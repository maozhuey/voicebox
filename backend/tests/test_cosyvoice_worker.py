"""Worker protocol tests that do not require a downloaded CosyVoice model."""

import queue

import numpy as np
import pytest

from backend.backends.cosyvoice_backend import (
    CosyVoiceSegmentTimeoutError,
    CosyVoiceTTSBackend,
    CosyVoiceWorkerStartupTimeoutError,
)


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


def test_ready_handshake_is_reused_for_two_requests(monkeypatch):
    events = queue.Queue()
    events.put({"kind": "ready"})
    backend = _configured_backend(events)
    submitted_ids = []

    def submit(request):
        submitted_ids.append(request["id"])
        events.put(
            {
                "kind": "result",
                "id": request["id"],
                "audio": np.array([len(submitted_ids)], dtype=np.float32),
                "sample_rate": 24000,
            }
        )

    monkeypatch.setattr(backend._worker_requests, "put", submit)
    monkeypatch.setattr("backend.config.get_model_load_timeout_seconds", lambda: 0.01)

    first, _ = backend._generate_in_worker("第一段。", {"ref_audio": "sample.wav"}, None, None)
    second, _ = backend._generate_in_worker("第二段。", {"ref_audio": "sample.wav"}, None, None)

    assert submitted_ids[0] != submitted_ids[1]
    assert first.tolist() == [1.0]
    assert second.tolist() == [2.0]
    assert backend._worker_ready is True


def test_worker_timeout_restart_clears_ready_state_and_requires_new_handshake(monkeypatch):
    initial_events = queue.Queue()
    initial_events.put({"kind": "ready"})
    backend = _configured_backend(initial_events)
    monkeypatch.setattr("backend.config.get_cosyvoice_segment_timeout_seconds", lambda: 0.01)

    with pytest.raises(CosyVoiceSegmentTimeoutError):
        backend._generate_in_worker("超时分段。", {"ref_audio": "sample.wav"}, None, None)

    worker = backend._worker
    backend._terminate_worker()

    assert worker.terminated is True
    assert backend._worker_ready is False

    events = queue.Queue()
    events.put({"kind": "ready"})
    backend._worker = FakeProcess()
    backend._worker_model_size = "rl"
    backend._worker_requests = queue.Queue()
    backend._worker_events = events

    def submit(request):
        events.put(
            {
                "kind": "result",
                "id": request["id"],
                "audio": np.array([0.3], dtype=np.float32),
                "sample_rate": 24000,
            }
        )

    monkeypatch.setattr(backend._worker_requests, "put", submit)
    audio, _ = backend._generate_in_worker("重建后。", {"ref_audio": "sample.wav"}, None, None)

    assert audio.tolist() == pytest.approx([0.3])
    assert backend._worker_ready is True


def test_worker_startup_has_a_separate_deadline(monkeypatch):
    backend = _configured_backend(queue.Queue())
    monkeypatch.setattr("backend.config.get_model_load_timeout_seconds", lambda: 0.01)
    monkeypatch.setattr("backend.config.get_cosyvoice_segment_timeout_seconds", lambda: 60.0)

    with pytest.raises(CosyVoiceWorkerStartupTimeoutError, match="startup"):
        backend._generate_in_worker("尚未就绪。", {"ref_audio": "sample.wav"}, None, None)


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
