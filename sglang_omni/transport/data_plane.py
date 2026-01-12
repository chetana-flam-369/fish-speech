# SPDX-License-Identifier: Apache-2.0
"""Data plane for tensor/data transfer via shared memory."""

import logging
import pickle
from multiprocessing import shared_memory as _shm
from typing import Any

from sglang_omni.core.types import SHMMetadata

logger = logging.getLogger(__name__)


def shm_write_bytes(payload: bytes) -> SHMMetadata:
    """Write bytes into SharedMemory and return metadata.

    Caller should close the segment; the receiver should unlink.
    """
    shm = _shm.SharedMemory(create=True, size=len(payload))
    mv = memoryview(shm.buf)
    mv[: len(payload)] = payload
    del mv
    meta = SHMMetadata(name=shm.name, size=len(payload))
    try:
        shm.close()
    except Exception as e:
        logger.debug("Failed to close shared memory: %s", e)
    return meta


def shm_read_bytes(meta: SHMMetadata) -> bytes:
    """Read bytes from SharedMemory by metadata and cleanup."""
    shm = _shm.SharedMemory(name=meta.name)
    mv = memoryview(shm.buf)
    data = bytes(mv[: meta.size])
    del mv
    try:
        shm.close()
    except Exception:
        pass
    try:
        shm.unlink()
    except Exception:
        pass
    return data


class SHMDataPlane:
    """Data plane using shared memory for inter-stage data transfer.

    Pattern:
    - put(): Serialize object, write to SHM, return metadata
    - get(): Read from SHM using metadata, deserialize, cleanup SHM
    - cleanup(): Force cleanup of SHM segment (for abort scenarios)
    """

    def __init__(self, threshold_bytes: int = 0):
        """Initialize SHM data plane.

        Args:
            threshold_bytes: Size threshold for using SHM vs inline.
                           0 means always use SHM (default for simplicity).
        """
        self.threshold = threshold_bytes
        self._pending_segments: dict[str, list[SHMMetadata]] = (
            {}
        )  # request_id -> [metadata]
        self._metrics = {
            "puts": 0,
            "gets": 0,
            "bytes_transferred": 0,
        }

    def put(
        self,
        request_id: str,
        data: Any,
        from_stage: str = "",
        to_stage: str = "",
    ) -> tuple[bool, SHMMetadata | None]:
        """Store data in shared memory.

        Args:
            request_id: Request identifier
            data: Python object to store
            from_stage: Source stage name (for logging)
            to_stage: Destination stage name (for logging)

        Returns:
            (success, metadata) tuple
        """
        try:
            payload = pickle.dumps(data)
            meta = shm_write_bytes(payload)

            # Track for potential cleanup
            if request_id not in self._pending_segments:
                self._pending_segments[request_id] = []
            self._pending_segments[request_id].append(meta)

            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += meta.size

            logger.debug(
                "SHM put: %s -> %s, req=%s, size=%d, shm=%s",
                from_stage,
                to_stage,
                request_id,
                meta.size,
                meta.name,
            )
            return True, meta

        except Exception as e:
            logger.error("SHM put failed for req %s: %s", request_id, e)
            return False, None

    def get(
        self,
        request_id: str,
        metadata: SHMMetadata,
        from_stage: str = "",
        to_stage: str = "",
    ) -> tuple[Any, int] | None:
        """Retrieve data from shared memory.

        Args:
            request_id: Request identifier
            metadata: SHM metadata from put operation
            from_stage: Source stage name (for logging)
            to_stage: Destination stage name (for logging)

        Returns:
            (object, size) tuple or None on failure
        """
        try:
            data_bytes = shm_read_bytes(metadata)
            obj = pickle.loads(data_bytes)

            # Remove from pending (it's been consumed)
            if request_id in self._pending_segments:
                self._pending_segments[request_id] = [
                    m
                    for m in self._pending_segments[request_id]
                    if m.name != metadata.name
                ]
                if not self._pending_segments[request_id]:
                    del self._pending_segments[request_id]

            self._metrics["gets"] += 1

            logger.debug(
                "SHM get: %s -> %s, req=%s, size=%d, shm=%s",
                from_stage,
                to_stage,
                request_id,
                metadata.size,
                metadata.name,
            )
            return obj, metadata.size

        except Exception as e:
            logger.error("SHM get failed for req %s: %s", request_id, e)
            return None

    def cleanup(self, request_id: str) -> None:
        """Force cleanup of all SHM segments for a request.

        Used when a request is aborted before data is consumed.
        """
        if request_id not in self._pending_segments:
            return

        for meta in self._pending_segments[request_id]:
            try:
                shm = _shm.SharedMemory(name=meta.name)
                shm.close()
                shm.unlink()
                logger.debug("SHM cleanup: req=%s, shm=%s", request_id, meta.name)
            except FileNotFoundError:
                # Already cleaned up
                pass
            except Exception as e:
                logger.warning(
                    "SHM cleanup failed for req %s, shm %s: %s",
                    request_id,
                    meta.name,
                    e,
                )

        del self._pending_segments[request_id]

    def health(self) -> dict[str, Any]:
        """Return health status and metrics."""
        return {
            "status": "healthy",
            "pending_requests": len(self._pending_segments),
            **self._metrics,
        }
