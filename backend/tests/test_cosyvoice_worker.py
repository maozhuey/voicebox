"""Worker protocol tests that do not require a downloaded CosyVoice model."""

import queue

import numpy as np
import pytest

from backend.backends.cosyvoice_backend import CosyVoiceSegmentTimeoutError, CosyVoiceTTSBackend


class FakeProcess:
    def __init__(self):
        self.alive = True
        self.terminated = False

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        return None

    def terminate(self):
        self.terminated = True
        self.alive = False

    def kill(self):
        self.alive = False


def _configured_backend(events):
    backend = CosyVoiceTTSBackend()
    backend._worker = FakeProcess()
    backend._worker_model_size = "rl"
    backend._worker_requests = queue.Queue()
    backend._worker_events = events
    return backend


def test_worker_protocol_reports_safe_phase_order_and_returns_audio(monkeypatch):
    events = queue.Queue()
    events.put({"kind": "ready"})
    events.put({"kind": "phase", "id": "ignored", "phase": "llm_decoding"})
    backend = _configured_backend(events)
    phases = []
    backend.set_phase_callback(phases.append)

    # The request ID is random; the fake event queue inserts matching messages
    # when the parent submits its request, as the real worker does.
    original_put = backend._worker_requests.put

    def submit(request):
        original_put(request)
        events.put({"kind": "phase", "id": request["id"], "phase": "preprocessing"})
        events.put({"kind": "phase", "id": request["id"], "phase": "llm_decoding"})
        events.put({"kind": "phase", "id": request["id"], "phase": "flow"})
        events.put({"kind": "phase", "id": request["id"], "phase": "vocoder"})
        events.put(
            {
                "kind": "result",
                "id": request["id"],
                "audio": np.array([0.1, 0.2], dtype=np.float32),
                "sample_rate": 24000,
            }
        )

    monkeypatch.setattr(backend._worker_requests, "put", submit)

    audio, sample_rate = backend._generate_in_worker("正文", {"ref_audio": "sample.wav"}, None, None)

    assert sample_rate == 24000
    assert np.allclose(audio, [0.1, 0.2])
    assert phases == ["preprocessing", "llm_decoding", "flow", "vocoder"]


def test_expired_worker_deadline_is_terminable(monkeypatch):
    events = queue.Queue()
    events.put({"kind": "ready"})
    backend = _configured_backend(events)
    monkeypatch.setattr(
        "backend.config.get_cosyvoice_segment_timeout_seconds", lambda: 0.01
    )

    with pytest.raises(CosyVoiceSegmentTimeoutError, match="preprocessing"):
        backend._generate_in_worker("无标点输入", {"ref_audio": "sample.wav"}, None, None)

    worker = backend._worker
    backend._terminate_worker()
    assert worker.terminated is True
