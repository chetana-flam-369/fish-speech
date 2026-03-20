# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible API server for sglang-omni.

Provides the following endpoints:
- POST /v1/chat/completions  — Text (+ audio) chat completions
- POST /v1/audio/speech      — Text-to-speech synthesis
- GET  /v1/models            — List available models
- GET  /v1/fs/list           — Browse filesystem directories
- GET  /v1/fs/file           — Download a file
- GET  /health               — Health check
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

from sglang_omni.client import (
    Client,
    ClientError,
    GenerateRequest,
    Message,
    SamplingParams,
)
from sglang_omni.serve.protocol import (
    ChatCompletionAudio,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamChoice,
    ChatCompletionStreamDelta,
    ChatCompletionStreamResponse,
    CreateSpeechRequest,
    ModelCard,
    ModelList,
    UsageResponse,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Streaming codec (lazy-loaded in the API process, decoded in thread pool)
#
# Loaded on CUDA so each decode step takes ~100-200 ms (GPU) instead of
# ~1700 ms (CPU), pushing warm TTFB well under 500 ms.
# max_workers=1: only one GPU decode at a time to avoid VRAM pressure.
# ---------------------------------------------------------------------------

_stream_codec_cache: dict = {}
_stream_codec_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dac_decode")
_stream_codec_lock: asyncio.Lock | None = None


def _get_stream_codec_lock() -> asyncio.Lock:
    global _stream_codec_lock
    if _stream_codec_lock is None:
        _stream_codec_lock = asyncio.Lock()
    return _stream_codec_lock


async def _get_stream_codec(model_name: str):
    """Return the CPU DAC codec, loading and warming it once in a background thread.

    We intentionally load on CPU rather than CUDA.  When the codec runs on the
    same GPU as the SGLang LLM, CUDA kernel scheduling contention degrades LLM
    TTFT by 70×+ (97 ms → 7+ s).  On CPU the warm decode costs ~180 ms per
    10-frame chunk, giving a warm TTFB of ~390 ms — much better than the
    7,500 ms observed with the GPU codec.
    """
    if "codec" in _stream_codec_cache:
        return _stream_codec_cache["codec"]

    async with _get_stream_codec_lock():
        if "codec" in _stream_codec_cache:
            return _stream_codec_cache["codec"]

        logger.info("[STREAM] Loading DAC codec on CPU (one-time) …")

        def _load():
            import os
            import torch
            from sglang_omni.models.fishaudio_s2_pro.pipeline.stages import (
                _load_codec,
                _resolve_checkpoint,
            )
            mp = os.environ.get("S2PRO_MODEL_PATH", model_name)
            ckpt = _resolve_checkpoint(mp)
            codec = _load_codec(ckpt, "cpu")
            # Warm-up decode: eliminates JIT/kernel-compilation cost on the
            # first real request (cold decode was ~1,750 ms; after warm-up ~180 ms).
            # Shape matches production: [batch=1, num_codebooks=10, chunk_tokens=10].
            dummy = torch.zeros(1, 10, 10, dtype=torch.long)
            try:
                with torch.no_grad():
                    codec.from_indices(dummy)
            except Exception:
                pass
            logger.info("[STREAM] DAC codec warm-up complete.")
            return codec

        loop = asyncio.get_event_loop()
        codec = await loop.run_in_executor(_stream_codec_executor, _load)
        _stream_codec_cache["codec"] = codec
        logger.info("[STREAM] DAC codec ready on CPU.")
        return codec


async def _decode_vq_chunk(codec, codebook_data: list) -> bytes:
    """Decode raw VQ codes → PCM bytes in a thread pool (non-blocking).

    The codec lives on CPU to avoid GPU contention with the LLM.  After
    warm-up a 10-frame chunk decodes in ~180 ms.
    """
    import numpy as np

    def _run():
        import torch
        # codebook_data: list[list[int]], shape [num_codebooks, CHUNK_TOKENS]
        codes = torch.tensor(codebook_data, dtype=torch.long)  # CPU
        codebook_input = codes.unsqueeze(0)  # [1, CB, T]
        with torch.no_grad():
            audio = codec.from_indices(codebook_input)  # [1, 1, T_audio]
        audio_np = audio[0, 0].float().numpy()
        return (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_stream_codec_executor, _run)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    client: Client,
    *,
    model_name: str | None = None,
) -> FastAPI:
    """Create a FastAPI application with OpenAI-compatible endpoints.

    Args:
        client: Client instance connected to the pipeline coordinator.
        model_name: Default model name to report in responses and /v1/models.
        serve_playground: Path to the playground directory to serve as static
            files.  When set, the filesystem browser API and static file
            serving are enabled so the entire playground runs on a single port.

    Returns:
        Configured FastAPI application.
    """
    app = FastAPI(title="sglang-omni", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store references in app state for access from route handlers
    app.state.client = client
    app.state.model_name = model_name or "sglang-omni"

    # Register all routes
    _register_health(app)
    _register_models(app)
    _register_chat_completions(app)
    _register_speech(app)

    return app


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


def _register_health(app: FastAPI) -> None:
    @app.get("/health")
    async def health() -> JSONResponse:
        """Health check endpoint (includes filesystem browse info)."""
        client: Client = app.state.client
        info = client.health()
        is_running = info.get("running", False)
        status_code = 200 if is_running else 503
        return JSONResponse(
            content={
                "status": "healthy" if is_running else "unhealthy",
                **info,
            },
            status_code=status_code,
        )


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------


def _register_models(app: FastAPI) -> None:
    @app.get("/v1/models")
    async def list_models() -> JSONResponse:
        """List available models."""
        model_name: str = app.state.model_name
        model_list = ModelList(
            data=[
                ModelCard(
                    id=model_name,
                    root=model_name,
                    created=0,
                )
            ]
        )
        return JSONResponse(content=model_list.model_dump())


# ---------------------------------------------------------------------------
# POST /v1/chat/completions
# ---------------------------------------------------------------------------


def _register_chat_completions(app: FastAPI) -> None:
    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest) -> Response:
        client: Client = app.state.client
        default_model: str = app.state.model_name

        request_id = req.request_id or str(uuid.uuid4())
        response_id = f"chatcmpl-{request_id}"
        created = int(time.time())
        model = req.model or default_model

        gen_req = _build_chat_generate_request(req)

        # Determine audio format from request
        audio_format = "wav"
        if req.audio and isinstance(req.audio, dict):
            audio_format = req.audio.get("format", "wav")

        if req.stream:
            return StreamingResponse(
                _chat_stream(
                    client,
                    gen_req,
                    request_id,
                    response_id,
                    created,
                    model,
                    req,
                    audio_format,
                ),
                media_type="text/event-stream",
            )

        return await _chat_non_stream(
            client,
            gen_req,
            request_id,
            response_id,
            created,
            model,
            req,
            audio_format,
        )


async def _chat_non_stream(
    client: Client,
    gen_req: GenerateRequest,
    request_id: str,
    response_id: str,
    created: int,
    model: str,
    req: ChatCompletionRequest,
    audio_format: str,
) -> JSONResponse:
    """Handle non-streaming chat completions."""
    try:
        result = await client.completion(
            gen_req,
            request_id=request_id,
            audio_format=audio_format,
        )
    except ClientError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Error generating response for request %s", request_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    requested_modalities = req.modalities or ["text"]

    # Build message content
    message: dict[str, Any] = {"role": "assistant"}

    if "text" in requested_modalities and result.text:
        message["content"] = result.text

    if "audio" in requested_modalities and result.audio is not None:
        message["audio"] = {
            "id": result.audio.id,
            "data": result.audio.data,
            "transcript": result.audio.transcript,
        }

    if "content" not in message and "audio" not in message:
        message["content"] = result.text

    # Build usage
    usage = None
    if result.usage is not None:
        usage = UsageResponse(
            prompt_tokens=result.usage.prompt_tokens or 0,
            completion_tokens=result.usage.completion_tokens or 0,
            total_tokens=result.usage.total_tokens or 0,
        )

    response = ChatCompletionResponse(
        id=response_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=message,
                finish_reason=result.finish_reason,
            )
        ],
        usage=usage,
    )

    return JSONResponse(content=response.model_dump())


async def _chat_stream(
    client: Client,
    gen_req: GenerateRequest,
    request_id: str,
    response_id: str,
    created: int,
    model: str,
    req: ChatCompletionRequest,
    audio_format: str,
):
    """Streaming chat completion generator (yields SSE events)."""
    role_sent = False
    requested_modalities = req.modalities or ["text"]
    finish_reason: str | None = None
    final_usage: UsageResponse | None = None

    async for chunk in client.completion_stream(
        gen_req,
        request_id=request_id,
        audio_format=audio_format,
    ):
        # Capture finish info for the dedicated finish chunk after the loop.
        if chunk.finish_reason is not None:
            finish_reason = chunk.finish_reason
            if chunk.usage is not None:
                final_usage = UsageResponse(
                    prompt_tokens=chunk.usage.prompt_tokens or 0,
                    completion_tokens=chunk.usage.completion_tokens or 0,
                    total_tokens=chunk.usage.total_tokens or 0,
                )
            continue

        delta = ChatCompletionStreamDelta()
        emit = False

        # Send role on first chunk
        if not role_sent:
            delta.role = "assistant"
            role_sent = True
            emit = True

        # Text chunk
        if chunk.modality == "text" and chunk.text and "text" in requested_modalities:
            delta.content = chunk.text
            emit = True

        # Audio chunk
        if (
            chunk.modality == "audio"
            and chunk.audio_b64 is not None
            and "audio" in requested_modalities
        ):
            delta.audio = ChatCompletionAudio(
                id=f"audio-{request_id}",
                data=chunk.audio_b64,
            )
            emit = True

        if not emit:
            continue

        stream_resp = ChatCompletionStreamResponse(
            id=response_id,
            created=created,
            model=model,
            choices=[
                ChatCompletionStreamChoice(
                    index=0,
                    delta=delta,
                    finish_reason=None,
                )
            ],
        )

        data = stream_resp.model_dump(exclude_none=True)
        for choice in data.get("choices", []):
            choice.setdefault("finish_reason", None)
        yield f"data: {json.dumps(data)}\n\n"

    # Finish chunk: empty delta + finish_reason.
    finish_resp = ChatCompletionStreamResponse(
        id=response_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionStreamChoice(
                index=0,
                delta=ChatCompletionStreamDelta(),
                finish_reason=finish_reason or "stop",
            )
        ],
        usage=final_usage,
    )
    data = finish_resp.model_dump(exclude_none=True)
    for choice in data.get("choices", []):
        choice.setdefault("finish_reason", None)
    yield f"data: {json.dumps(data)}\n\n"

    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Request building helpers
# ---------------------------------------------------------------------------


def _build_chat_generate_request(req: ChatCompletionRequest) -> GenerateRequest:
    """Convert a ChatCompletionRequest into a client GenerateRequest."""
    # Parse stop sequences
    stop: list[str] = []
    if isinstance(req.stop, str):
        stop = [req.stop]
    elif isinstance(req.stop, list):
        stop = list(req.stop)

    # Build sampling params
    sampling = SamplingParams(
        temperature=req.temperature if req.temperature is not None else 1.0,
        top_p=req.top_p if req.top_p is not None else 1.0,
        top_k=req.top_k if req.top_k is not None else -1,
        min_p=req.min_p if req.min_p is not None else 0.0,
        repetition_penalty=(
            req.repetition_penalty if req.repetition_penalty is not None else 1.0
        ),
        stop=stop,
        seed=req.seed,
        max_new_tokens=req.effective_max_tokens,
    )

    # Convert messages
    messages = [Message(role=m.role, content=m.content) for m in req.messages]

    # Determine output modalities
    output_modalities = req.modalities  # e.g. ["text", "audio"]

    # Build per-stage sampling overrides
    stage_sampling: dict[str, SamplingParams] | None = None
    if req.stage_sampling:
        stage_sampling = {}
        for stage_name, params_dict in req.stage_sampling.items():
            stage_sampling[stage_name] = SamplingParams(**params_dict)

    # Extract audios, images, and videos from request
    audios: list[str] | None = None
    if req.audios:
        audios = req.audios

    images: list[str] | None = None
    if req.images:
        images = req.images

    videos: list[str] | None = None
    if req.videos:
        videos = req.videos

    # Merge audio config, audios, images, and videos into metadata
    metadata: dict[str, Any] = {}
    if req.audio:
        metadata["audio_config"] = req.audio
    if audios:
        metadata["audios"] = audios
    if images:
        metadata["images"] = images
    if videos:
        metadata["videos"] = videos

    return GenerateRequest(
        model=req.model,
        messages=messages,
        sampling=sampling,
        stage_sampling=stage_sampling,
        stage_params=req.stage_params,
        stream=req.stream,
        max_tokens=req.effective_max_tokens,
        output_modalities=output_modalities,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# POST /v1/audio/speech
# ---------------------------------------------------------------------------


def _register_speech(app: FastAPI) -> None:
    @app.post("/v1/audio/speech")
    async def create_speech(req: CreateSpeechRequest) -> Response:
        client: Client = app.state.client
        default_model: str = app.state.model_name

        request_id = f"speech-{uuid.uuid4()}"

        gen_req = _build_speech_generate_request(req, default_model)

        try:
            result = await client.speech(
                gen_req,
                request_id=request_id,
                response_format=req.response_format,
                speed=req.speed,
            )
        except ClientError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Error generating speech for request %s", request_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return Response(
            content=result.audio_bytes,
            media_type=result.mime_type,
            headers={
                "Content-Disposition": f'attachment; filename="speech.{result.format}"',
            },
        )

    @app.post("/v1/audio/speech/stream")
    async def create_speech_stream(req: CreateSpeechRequest) -> StreamingResponse:
        """True streaming speech endpoint.

        Architecture
        ------------
        The tts_engine worker and this API handler share the same asyncio
        event loop (same OS process).  VQ codes are therefore passed via an
        in-process asyncio.Queue (_vq_streaming_queues) rather than through
        the ZMQ/msgpack control-plane channel, which would silently drop
        large list[list[int]] payloads on serialisation errors.

        The pipeline is still driven by running client.generate() in a
        background asyncio task so that the coordinator/scheduler stay alive.
        Audio chunks are yielded as soon as they are decoded, giving true
        streaming TTFB.
        """
        import struct
        import time as _time

        from sglang_omni.models.fishaudio_s2_pro.pipeline.stages import (
            _vq_streaming_queues,
        )

        client_inner: Client = app.state.client
        default_model_inner: str = app.state.model_name
        request_id = f"speech-stream-{uuid.uuid4()}"

        base_req = _build_speech_generate_request(req, default_model_inner)
        gen_req = GenerateRequest(
            model=base_req.model,
            prompt=base_req.prompt,
            sampling=base_req.sampling,
            stage_params=base_req.stage_params,
            stream=True,
            output_modalities=base_req.output_modalities,
            metadata=base_req.metadata,
        )

        sample_rate = 44100
        t_start = _time.perf_counter()
        first_chunk_sent = False

        # Pre-load codec so first decode is fast (no cold-start delay).
        codec = await _get_stream_codec(default_model_inner)

        # Register the in-process queue BEFORE the pipeline starts so that
        # _stream_builder can put_nowait() as soon as the first token arrives.
        vq_queue: asyncio.Queue = asyncio.Queue()
        _vq_streaming_queues[request_id] = vq_queue

        async def audio_generator():
            nonlocal first_chunk_sent

            # WAV header with streaming-size placeholders (0xFFFFFFFF).
            header = struct.pack(
                "<4sI4s4sIHHIIHH4sI",
                b"RIFF", 0xFFFFFFFF, b"WAVE",
                b"fmt ", 16, 1, 1,
                sample_rate, sample_rate * 2, 2, 16,
                b"data", 0xFFFFFFFF,
            )
            yield header

            async def _drain_pipeline() -> None:
                """Drive the pipeline to completion (coordinator + scheduler
                need the generate() loop to be consumed)."""
                try:
                    async for _ in client_inner.generate(
                        gen_req, request_id=request_id
                    ):
                        pass
                except Exception as exc:
                    logger.exception(
                        "[STREAM] Pipeline error for %s: %s", request_id, exc
                    )

            pipeline_task = asyncio.create_task(_drain_pipeline())

            try:
                while True:
                    try:
                        item = await asyncio.wait_for(vq_queue.get(), timeout=30.0)
                    except asyncio.TimeoutError:
                        logger.warning(
                            "[STREAM] VQ queue timed out for %s", request_id
                        )
                        break

                    if item is None:  # sentinel: _result_builder signals end-of-stream
                        break

                    try:
                        pcm = await _decode_vq_chunk(codec, item)
                        if not first_chunk_sent:
                            ttfb_ms = (_time.perf_counter() - t_start) * 1000
                            logger.info(
                                "[TIMING] /stream real TTFB (first audio to client): %.1f ms",
                                ttfb_ms,
                            )
                            first_chunk_sent = True
                        yield pcm
                    except Exception as exc:
                        logger.exception(
                            "[STREAM] Decode error for %s: %s", request_id, exc
                        )

            finally:
                # Remove queue so _stream_builder stops putting new items.
                _vq_streaming_queues.pop(request_id, None)
                # Wait for the pipeline to finish (vocoder stage may still run).
                try:
                    await asyncio.wait_for(pipeline_task, timeout=10.0)
                except (asyncio.TimeoutError, asyncio.CancelledError, Exception) as e:
                    logger.debug("[STREAM] Pipeline cleanup for %s: %s", request_id, e)
                    pipeline_task.cancel()

            if not first_chunk_sent:
                logger.warning(
                    "[STREAM] No streaming chunks received for %s", request_id
                )

        return StreamingResponse(
            audio_generator(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'attachment; filename="speech_stream.wav"',
                "Transfer-Encoding": "chunked",
                "X-Request-Id": request_id,
            },
        )


def _build_speech_generate_request(
    req: CreateSpeechRequest,
    default_model: str,
) -> GenerateRequest:
    """Convert a CreateSpeechRequest into a client GenerateRequest."""

    # Build TTS-specific parameters to pass through the pipeline
    tts_params: dict[str, Any] = {
        "voice": req.voice,
        "response_format": req.response_format,
        "speed": req.speed,
    }
    if req.task_type is not None:
        tts_params["task_type"] = req.task_type
    if req.language is not None:
        tts_params["language"] = req.language
    if req.instructions is not None:
        tts_params["instructions"] = req.instructions
    if req.ref_audio is not None:
        tts_params["ref_audio"] = req.ref_audio
    if req.ref_text is not None:
        tts_params["ref_text"] = req.ref_text
    if req.seed is not None:
        tts_params["seed"] = req.seed

    # Sampling params — use S2-Pro-tuned defaults
    sampling = SamplingParams(
        temperature=0.8, top_p=0.8, top_k=30, repetition_penalty=1.1
    )
    if req.max_new_tokens is not None:
        sampling.max_new_tokens = req.max_new_tokens
    if req.temperature is not None:
        sampling.temperature = req.temperature
    if req.top_p is not None:
        sampling.top_p = req.top_p
    if req.top_k is not None:
        sampling.top_k = req.top_k
    if req.repetition_penalty is not None:
        sampling.repetition_penalty = req.repetition_penalty

    # Build prompt: plain string if no references, dict otherwise
    prompt: Any = req.input
    if req.ref_audio is not None:
        ref = {"audio_path": req.ref_audio}
        if req.ref_text is not None:
            ref["text"] = req.ref_text
        prompt = {"text": req.input, "references": [ref]}

    return GenerateRequest(
        model=req.model or default_model,
        prompt=prompt,
        sampling=sampling,
        stage_params=req.stage_params,
        stream=False,  # TTS returns complete audio, no streaming
        output_modalities=["audio"],
        metadata={
            "task": "tts",
            "tts_params": tts_params,
        },
    )
