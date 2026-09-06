"""TTS runtime lease tests for safe CosyVoice memory switching."""

import asyncio

import pytest


class FakeBackend:
    def __init__(self, loaded: bool = True):
        self.loaded = loaded
        self.unload_count = 0

    def is_loaded(self) -> bool:
        return self.loaded

    def unload_model(self) -> None:
        self.unload_count += 1
        self.loaded = False


@pytest.fixture(autouse=True)
def reset_runtime_coordinator(monkeypatch):
    import backend.backends as backends

    monkeypatch.setattr(backends, "_tts_backends", {})
    monkeypatch.setattr(backends, "_active_tts_leases", {})
    monkeypatch.setattr(backends, "_tts_runtime_condition", asyncio.Condition())


@pytest.mark.asyncio
async def test_cosyvoice_releases_idle_qwen_before_it_acquires_runtime():
    import backend.backends as backends

    qwen = FakeBackend()
    cosy = FakeBackend()
    backends._tts_backends.update({"qwen": qwen, "cosyvoice": cosy})

    async with backends.acquire_tts_runtime("cosyvoice"):
        assert qwen.unload_count == 1
        assert cosy.unload_count == 0


@pytest.mark.asyncio
async def test_cosyvoice_waits_for_an_active_qwen_lease_instead_of_unloading_it():
    import backend.backends as backends

    qwen = FakeBackend()
    backends._tts_backends["qwen"] = qwen
    cosy_entered = asyncio.Event()

    async with backends.acquire_tts_runtime("qwen"):
        waiter = asyncio.create_task(_acquire_cosyvoice(backends, cosy_entered))
        await asyncio.sleep(0)
        assert not cosy_entered.is_set()
        assert qwen.unload_count == 0

    await waiter
    assert cosy_entered.is_set()
    assert qwen.unload_count == 1


@pytest.mark.asyncio
async def test_qwen_releases_an_idle_cosyvoice_worker_before_loading():
    import backend.backends as backends

    qwen = FakeBackend()
    cosy = FakeBackend()
    backends._tts_backends.update({"qwen": qwen, "cosyvoice": cosy})

    async with backends.acquire_tts_runtime("qwen"):
        assert cosy.unload_count == 1
        assert qwen.unload_count == 0


async def _acquire_cosyvoice(backends, entered: asyncio.Event) -> None:
    async with backends.acquire_tts_runtime("cosyvoice"):
        entered.set()
