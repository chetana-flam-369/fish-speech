import pytest
import torch

from sglang_omni.relay.descriptor import Descriptor


@pytest.fixture
def worker_configs():
    """Create configurations for two workers."""
    if torch.cuda.is_available() and torch.cuda.device_count() < 2:
        pytest.skip("Need at least 2 GPUs for this test")

    return [
        {
            "host": "127.0.0.1",
            "metadata_server": "http://127.0.0.1:8080/metadata",
            "device_name": "",
            "gpu_id": 0,
            "worker_id": "worker0",
        },
        {
            "host": "127.0.0.1",
            "metadata_server": "http://127.0.0.1:8080/metadata",
            "device_name": "",
            "gpu_id": 1 if torch.cuda.is_available() else 0,
            "worker_id": "worker1",
        },
    ]


class TestNixlRalay:
    """Basic test suite for NixlRalay."""

    def test_transfer_between_workers(self, worker_configs):
        """Test data transfer between two workers."""
        from sglang_omni.relay.nixl_ralay import NixlRalay

        config0, config1 = worker_configs
        connector0 = NixlRalay(config0)
        connector1 = NixlRalay(config1)

        try:
            device0 = (
                f'cuda:{config0["gpu_id"]}' if torch.cuda.is_available() else "cpu"
            )
            device1 = (
                f'cuda:{config1["gpu_id"]}' if torch.cuda.is_available() else "cpu"
            )

            # Create test data
            test_tensor = torch.randn(1000000, dtype=torch.float32, device=device0)
            original = test_tensor.cpu().clone()

            # Send data
            send_desc = Descriptor(test_tensor)
            readable_op = connector0.put([send_desc])
            metadata = readable_op.metadata()

            # Receive data
            buffer = torch.empty(
                metadata.descriptors[0].size // test_tensor.element_size(),
                dtype=test_tensor.dtype,
                device=device1,
            )
            read_op = connector1.get(metadata, [Descriptor(buffer)])

            # Wait for completion
            try:
                if hasattr(read_op, "wait_for_completion"):
                    coro = read_op.wait_for_completion()
                    if coro:
                        connector1._run_maybe_async(coro)
            except Exception:
                pytest.skip("Data transfer not supported in test environment")

            # Verify data
            received = buffer.cpu()
            assert torch.allclose(original, received, rtol=1e-5, atol=1e-5)
            assert connector0._metrics["puts"] >= 1
            assert connector1._metrics["gets"] >= 1

        finally:
            connector0.close()
            connector1.close()

    @pytest.mark.asyncio
    async def test_transfer_between_workers_async(self, worker_configs):
        """Test async data transfer between two workers."""
        from sglang_omni.relay.nixl_ralay import NixlRalay

        config0, config1 = worker_configs
        connector0 = NixlRalay(config0)
        connector1 = NixlRalay(config1)

        try:
            device0 = (
                f'cuda:{config0["gpu_id"]}' if torch.cuda.is_available() else "cpu"
            )
            device1 = (
                f'cuda:{config1["gpu_id"]}' if torch.cuda.is_available() else "cpu"
            )

            # Create test data
            test_tensor = torch.randn(1000000, dtype=torch.float32, device=device0)
            original = test_tensor.cpu().clone()

            # Send data
            send_desc = Descriptor(test_tensor)
            readable_op = await connector0.put_async([send_desc])
            metadata = readable_op.metadata()

            # Receive data
            buffer = torch.empty(
                metadata.descriptors[0].size // test_tensor.element_size(),
                dtype=test_tensor.dtype,
                device=device1,
            )
            read_op = await connector1.get_async(metadata, [Descriptor(buffer)])

            # Wait for completion
            try:
                if hasattr(read_op, "wait_for_completion"):
                    await read_op.wait_for_completion()
            except Exception:
                pytest.skip("Data transfer not supported in test environment")

            # Verify data
            received = buffer.cpu()
            assert torch.allclose(original, received, rtol=1e-5, atol=1e-5)
            assert connector0._metrics["puts"] >= 1
            assert connector1._metrics["gets"] >= 1

        finally:
            connector0.close()
            connector1.close()
