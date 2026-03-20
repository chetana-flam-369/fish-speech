# SPDX-License-Identifier: Apache-2.0
"""Stage executor factories for the S2-Pro TTS pipeline."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

import torch

from sglang_omni.executors import EngineExecutor, PreprocessingExecutor
from sglang_omni.models.fishaudio_s2_pro.io import S2ProState
from sglang_omni.models.fishaudio_s2_pro.pipeline.engine_io import (
    apply_tts_result,
    build_sglang_tts_request,
)
from sglang_omni.models.fishaudio_s2_pro.pipeline.state_io import (
    load_state,
    store_state,
)
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-process streaming queues (bypasses ZMQ/msgpack for VQ code payloads)
#
# The tts_engine worker and the API server share the same asyncio event loop
# (same process).  Rather than serialising large list[list[int]] arrays
# through msgpack → ZMQ → msgpack, we drop the payload directly into an
# asyncio.Queue that the /stream endpoint consumes.
#
# Protocol:
#   - Each item is  list[list[int]]  (shape [num_codebooks, CHUNK_TOKENS])
#   - A  None  sentinel signals end-of-stream.
# ---------------------------------------------------------------------------
_vq_streaming_queues: dict[str, asyncio.Queue] = {}


# ---------------------------------------------------------------------------
# Helpers (model loading)
# ---------------------------------------------------------------------------


def _resolve_checkpoint(checkpoint: str) -> str:
    if os.path.isdir(checkpoint):
        return checkpoint
    from huggingface_hub import snapshot_download

    return snapshot_download(checkpoint)


def _load_audio_decoder(checkpoint: str, device: str):
    from transformers import PreTrainedTokenizerFast

    from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic.configuration import (
        FishQwen3OmniConfig,
    )
    from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic.modeling import (
        FishQwen3OmniForCausalLM,
    )

    checkpoint = _resolve_checkpoint(checkpoint)
    logger.info("Loading S2-Pro model from %s …", checkpoint)
    t0 = time.perf_counter()

    config = FishQwen3OmniConfig.from_pretrained(checkpoint)

    # Load directly in bfloat16 to avoid a float32→bfloat16 conversion step
    # (halves peak CPU RAM).  Do NOT use low_cpu_mem_usage=True — with some
    # versions of accelerate it can silently dispatch layers to the GPU even
    # without device_map, which exhausts VRAM before SGLang starts.
    model = FishQwen3OmniForCausalLM.from_pretrained(
        checkpoint,
        config=config,
        torch_dtype=torch.bfloat16,
    )
    model.eval()

    # Extract the audio decoder sub-module and move only it to the target device.
    # The rest of the full model (text transformer) is immediately freed so that
    # SGLang can claim maximum VRAM for the KV cache.
    audio_decoder = model.audio_decoder
    audio_decoder.to(device=device)
    num_codebooks = config.audio_decoder_config.num_codebooks
    codebook_size = config.audio_decoder_config.vocab_size

    del model
    torch.cuda.empty_cache()
    logger.info("Audio decoder loaded in %.2fs", time.perf_counter() - t0)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint)
    return audio_decoder, num_codebooks, codebook_size, tokenizer, checkpoint


def _load_codec(checkpoint_dir: str, device: str):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)

    codec_path = os.path.join(checkpoint_dir, "codec.pth")
    logger.info("Loading DAC codec from %s …", codec_path)
    t0 = time.perf_counter()

    import sglang_omni.models.fishaudio_s2_pro.fish_speech.models.dac.modded_dac as _dac_mod

    configs_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(_dac_mod.__file__))),
        "configs",
    )
    cfg = OmegaConf.load(os.path.join(configs_dir, "modded_dac_vq.yaml"))
    codec = instantiate(cfg)

    state_dict = torch.load(
        codec_path, map_location=device, mmap=True, weights_only=True
    )
    codec.load_state_dict(state_dict, strict=False, assign=True)
    codec.eval()
    codec.to(device)
    logger.info("DAC codec loaded in %.2fs", time.perf_counter() - t0)
    return codec


# ---------------------------------------------------------------------------
# Stage 1: Preprocessing
# ---------------------------------------------------------------------------


def create_preprocessing_executor(model_path: str) -> PreprocessingExecutor:
    checkpoint_dir = _resolve_checkpoint(model_path)

    from transformers import PreTrainedTokenizerFast

    from sglang_omni.models.fishaudio_s2_pro.tokenizer import (
        Reference,
        S2ProTokenizerAdapter,
    )

    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint_dir)
    adapter = S2ProTokenizerAdapter(tokenizer)

    # Lazy-loaded codec
    _codec_cache: dict[str, Any] = {}

    def _get_codec(device: str = "cpu"):
        if "codec" not in _codec_cache:
            _codec_cache["codec"] = _load_codec(checkpoint_dir, device)
        return _codec_cache["codec"]

    def _encode_reference_audio(audio_path: str, device: str = "cpu") -> torch.Tensor:
        import torchaudio

        codec = _get_codec(device)
        audio, sr = torchaudio.load(audio_path)
        if audio.shape[0] > 1:
            audio = audio.mean(0, keepdim=True)
        audio = torchaudio.functional.resample(audio, sr, codec.sample_rate)
        # s2-pro-alpha codec expects [B, T] (adds channel dim internally)
        audios = audio.squeeze(0).unsqueeze(0).to(device)  # [1, T]
        audio_lengths = torch.tensor([audios.shape[1]], device=device, dtype=torch.long)
        with torch.no_grad():
            indices, _ = codec.encode(audios, audio_lengths)
            if indices.ndim == 3:
                indices = indices[0]
        return indices.cpu()

    def _preprocess(payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs or {}
        params = payload.request.params or {}

        # Speech endpoint sends prompt as a plain string
        if isinstance(inputs, str):
            inputs = {"text": inputs}

        text = inputs.get("text", "")
        num_codebooks = inputs.get("num_codebooks", 10)
        codebook_size = inputs.get("codebook_size", 4096)

        # Build voice-cloning references
        references: list[Reference] | None = None
        raw_refs = inputs.get("references")
        if raw_refs:
            references = []
            for ref_data in raw_refs:
                vq_codes = ref_data.get("vq_codes")
                if vq_codes is not None and not isinstance(vq_codes, torch.Tensor):
                    vq_codes = torch.tensor(vq_codes)

                if vq_codes is None and ref_data.get("audio_path"):
                    vq_codes = _encode_reference_audio(ref_data["audio_path"])

                references.append(
                    Reference(
                        audio_bytes=b"",
                        text=ref_data.get("text", ""),
                        vq_codes=vq_codes,
                    )
                )

        prompt_data = adapter.build_prompt(
            text=text,
            references=references,
            num_codebooks=num_codebooks,
        )

        state = S2ProState(
            input_ids=prompt_data["input_ids"],
            vq_mask_tokens=prompt_data["vq_mask_tokens"],
            vq_parts=prompt_data["vq_parts"],
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            max_new_tokens=params.get("max_new_tokens", 1024),
            temperature=params.get("temperature", 0.8),
            top_p=params.get("top_p", 0.8),
            top_k=params.get("top_k", 30),
            repetition_penalty=params.get("repetition_penalty", 1.1),
        )
        return store_state(payload, state)

    return PreprocessingExecutor(_preprocess)


# ---------------------------------------------------------------------------
# Stage 2: TTS Engine (S2-Pro)
# ---------------------------------------------------------------------------


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda",
    max_new_tokens: int = 2048,
    top_k: int = 30,
) -> EngineExecutor:
    """Factory for the S2-Pro TTS engine stage."""
    from sglang.srt.server_args import ServerArgs

    from sglang_omni.models.fishaudio_s2_pro.factory import (
        _patch_fish_config_for_sglang,
        create_s2pro_sglang_engine,
    )

    audio_decoder, num_codebooks, codebook_size, tokenizer, checkpoint_dir = (
        _load_audio_decoder(model_path, device)
    )
    audio_decoder.setup_caches(max_batch_size=1, dtype=torch.bfloat16)

    _patch_fish_config_for_sglang(checkpoint_dir)
    server_args = ServerArgs(
        model_path=checkpoint_dir,
        tp_size=1,
        dtype="bfloat16",
        mem_fraction_static=0.70,
        chunked_prefill_size=8192,
        max_running_requests=64,
        disable_cuda_graph=True,
    )

    engine = create_s2pro_sglang_engine(
        server_args=server_args,
        audio_decoder=audio_decoder,
        tokenizer=tokenizer,
        gpu_id=int(device.split(":")[-1]) if ":" in device else 0,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
    )

    # -----------------------------------------------------------------------
    # Streaming: emit raw VQ code chunks every CHUNK_TOKENS decode steps.
    #
    # item from _stream_adapter (factory.py) is step_out.codes:
    #   CUDA tensor of shape [num_codebooks+1, 1] per decode step.
    # Row 0 = semantic token (skip). Rows 1..num_codebooks = DAC codes.
    #
    # We do NOT call the DAC codec here — that would block the asyncio
    # event loop and stall the LLM.  Instead we return raw integer codes
    # (modality="vq_codes") so the API process can decode them in a
    # thread pool without touching the GPU event loop.
    # -----------------------------------------------------------------------
    _CHUNK_TOKENS = 10  # emit every 10 tokens ≈ 116 ms of audio

    _req_buffers: dict[str, list] = {}   # per-request list of CPU code tensors
    _req_ttft_done: set[str] = set()     # requests whose TTFT was already logged
    _req_tstart: dict[str, float] = {}   # request start timestamps

    def _request_builder(payload: StagePayload):
        _req_tstart[payload.request_id] = time.perf_counter()
        state = load_state(payload)
        return build_sglang_tts_request(state, tokenizer)

    def _stream_builder(payload: StagePayload, item: Any) -> Any:
        """Called once per LLM decode step.  item = CUDA tensor [CB+1, 1].

        Accumulates CHUNK_TOKENS frames, then sends the batch directly to the
        API process via an in-process asyncio.Queue — no ZMQ/msgpack involved
        for the actual payload, which avoids silent serialisation failures.

        Returns {} (empty heartbeat) through the ZMQ path every step so the
        coordinator still sees stream events and stays alive.
        """
        if item is None:
            return {}

        req_id = payload.request_id

        # Move to CPU — fast memcpy, does not block event loop.
        try:
            codes_cpu = item.detach().cpu()  # [num_codebooks+1, 1]
        except Exception as exc:
            logger.warning("[STREAM] Failed to move codes to CPU: %s", exc)
            return {}

        # Log TTFT exactly once per request.
        if req_id not in _req_ttft_done:
            t0 = _req_tstart.get(req_id, time.perf_counter())
            logger.info(
                "[TIMING] TTFT (sglang prefill→first token): %.1f ms",
                (time.perf_counter() - t0) * 1000,
            )
            _req_ttft_done.add(req_id)

        # Accumulate frames.
        buf = _req_buffers.setdefault(req_id, [])
        buf.append(codes_cpu)

        if len(buf) < _CHUNK_TOKENS:
            return {}

        # Flush CHUNK_TOKENS frames into the in-process queue.
        chunk_codes = buf[:_CHUNK_TOKENS]
        _req_buffers[req_id] = buf[_CHUNK_TOKENS:]

        # Stack: [num_codebooks+1, CHUNK_TOKENS] → skip row 0 (semantic).
        stacked = torch.cat(chunk_codes, dim=1)   # [num_codebooks+1, CHUNK_TOKENS]
        codebook_data = stacked[1:].tolist()       # list[list[int]]

        q = _vq_streaming_queues.get(req_id)
        if q is not None:
            q.put_nowait(codebook_data)
        else:
            logger.debug("[STREAM] No in-process queue for req %s; chunk dropped", req_id)

        # Return empty dict — just a lightweight ZMQ heartbeat, no payload.
        return {}

    def _result_builder(payload: StagePayload, result: Any) -> StagePayload:
        """Clean up per-request streaming state after LLM finishes."""
        req_id = payload.request_id

        # Flush any remaining tokens that didn't fill a full chunk.
        buf = _req_buffers.get(req_id, [])
        if buf:
            stacked = torch.cat(buf, dim=1)
            remaining = stacked[1:].tolist()
            q = _vq_streaming_queues.get(req_id)
            if q is not None:
                q.put_nowait(remaining)

        # Signal end-of-stream to the API process.
        q = _vq_streaming_queues.get(req_id)
        if q is not None:
            q.put_nowait(None)

        _req_buffers.pop(req_id, None)
        _req_ttft_done.discard(req_id)
        _req_tstart.pop(req_id, None)
        state = load_state(payload)
        apply_tts_result(state, result)
        return store_state(payload, state)

    return EngineExecutor(
        engine=engine,
        request_builder=_request_builder,
        result_builder=_result_builder,
        stream_builder=_stream_builder,
    )


# ---------------------------------------------------------------------------
# Stage 3: Vocoder (DAC codec decode)
# ---------------------------------------------------------------------------


def create_vocoder_executor(
    model_path: str,
    *,
    device: str = "cuda",
) -> PreprocessingExecutor:
    """Factory for the vocoder stage."""
    checkpoint_dir = _resolve_checkpoint(model_path)
    codec = _load_codec(checkpoint_dir, device)

    def _vocode(payload: StagePayload) -> StagePayload:
        state = load_state(payload)
        output_codes = state.output_codes

        codebook_codes = output_codes[1:].to(device)

        with torch.no_grad():
            audio = codec.from_indices(codebook_codes[None])

        audio_np = audio[0, 0].float().cpu()
        state.audio_samples = audio_np
        state.sample_rate = codec.sample_rate
        payload = store_state(payload, state)

        payload.data["audio_data"] = audio_np.tolist()
        payload.data["sample_rate"] = codec.sample_rate
        payload.data["modality"] = "audio"
        return payload

    return PreprocessingExecutor(_vocode)
