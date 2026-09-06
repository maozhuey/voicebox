"""CosyVoice 3 TTS backend for zero-shot voice cloning and instructed reading.

CosyVoice is intentionally kept as vendored source instead of a normal Python
dependency: upstream does not publish an inference package to PyPI.  Setup
installs the source at one audited revision under ``backend/vendors``; frozen
builds bundle the same source tree next to the executable.
"""

from __future__ import annotations

import asyncio
import gc
import importlib.util
import logging
import multiprocessing
import os
import queue
import sys
import tempfile
import threading
import time
import types
from pathlib import Path
from typing import ClassVar

import numpy as np

from . import TTSBackend
from .base import (
    combine_voice_prompts as _combine_voice_prompts,
    empty_device_cache,
    get_torch_device,
    is_model_cached,
    model_load_progress,
)

logger = logging.getLogger(__name__)

COSYVOICE_HF_REPOS = {
    # Upstream ships the RL checkpoint as ``llm.rl.pt`` inside the public base
    # repository, not as a separate HuggingFace repository.
    "rl": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "base": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
}

# The backend cannot successfully construct a model when only a partial Hub
# snapshot exists. These are the common inference artifacts for both variants.
_REQUIRED_FILES = ["flow.pt", "hift.pt", "cosyvoice3.yaml", "campplus.onnx", "speech_tokenizer_v3.onnx"]
_SAMPLE_RATE = 24_000
_SYSTEM_PROMPT = "You are a helpful assistant."
_END_OF_PROMPT = "<|endofprompt|>"
_MIN_VALID_AUDIO_SECONDS = 0.2
_MAX_SPEECH_ZERO_CROSSING_RATE = 0.25
_QWEN2_COMPAT_MODULE = "transformers.models.qwen2.modeling_qwen2_440"
_COSYVOICE_PHASES = frozenset({"preprocessing", "llm_decoding", "flow", "vocoder"})


class CosyVoiceSegmentTimeoutError(RuntimeError):
    """A worker was terminated after one independently bounded segment."""

    def __init__(self, phase: str):
        self.phase = phase
        super().__init__(f"CosyVoice generation timed out during {phase}")


class CosyVoiceWorkerError(RuntimeError):
    """A private worker exited without a usable audio result."""

    def __init__(self, phase: str):
        self.phase = phase
        super().__init__(f"CosyVoice generation failed during {phase}")


def _cosyvoice_worker_main(requests, events, model_size: str) -> None:
    """Own the upstream model and all of its non-cancellable inner threads."""
    backend = CosyVoiceTTSBackend()
    try:
        backend._load_model_sync(model_size)
        events.put({"kind": "ready"})
        while True:
            request = requests.get()
            if request is None:
                return
            request_id = request["id"]
            phase = "preprocessing"

            def report(next_phase: str, *, event_request_id: str = request_id) -> None:
                nonlocal phase
                phase = next_phase
                events.put({"kind": "phase", "id": event_request_id, "phase": phase})

            try:
                audio, sample_rate = backend._generate_sync(
                    request["text"],
                    request["voice_prompt"],
                    request["seed"],
                    request["instruct"],
                    report,
                )
                events.put(
                    {
                        "kind": "result",
                        "id": request_id,
                        "audio": audio,
                        "sample_rate": sample_rate,
                    }
                )
            except Exception:
                # The parent deliberately receives no traceback, text, prompt
                # path or upstream error.  This stable phase is enough to make
                # a history entry actionable without leaking user data.
                events.put({"kind": "error", "id": request_id, "phase": phase})
    except Exception:
        events.put({"kind": "startup_error"})
    finally:
        backend.unload_model()


def _format_instruct_prompt(instruct: str) -> str:
    """Build one CosyVoice 3 control section without leaking it into speech."""
    # 朗读指令必须位于唯一的结束标记之前。旧实现把零样本参考文本的结束标记
    # 放在指令之前, 会让模型把“请用河南话表达”等控制文字当成正文朗读。
    human_instruct = instruct.replace(_END_OF_PROMPT, "").strip()
    if human_instruct.startswith(_SYSTEM_PROMPT):
        human_instruct = human_instruct[len(_SYSTEM_PROMPT) :].strip()
    return f"{_SYSTEM_PROMPT} {human_instruct}{_END_OF_PROMPT}"


def _format_reference_prompt(reference_text: str) -> str:
    """Separate the zero-shot system prompt from the spoken reference text."""
    if _END_OF_PROMPT in reference_text:
        return reference_text
    return f"{_SYSTEM_PROMPT}{_END_OF_PROMPT}{reference_text}"


def _validate_generated_audio(audio: np.ndarray, sample_rate: int, text: str) -> None:
    """Reject known early-stop and broadband-noise failures."""
    if len(text.strip()) >= 8 and len(audio) / sample_rate < _MIN_VALID_AUDIO_SECONDS:
        raise RuntimeError(
            "CosyVoice 3 stopped before producing usable speech. Please retry the generation."
        )

    frame_length = max(1, int(sample_rate * 0.02))
    frame_count = len(audio) // frame_length
    if frame_count < 10:
        return

    frames = np.asarray(audio[: frame_count * frame_length], dtype=np.float32).reshape(
        frame_count,
        frame_length,
    )
    rms = np.sqrt(np.mean(frames**2, axis=1))
    active_frames = frames[rms >= 0.01]
    if len(active_frames) < 10:
        return

    # Normal voiced/unvoiced speech stays well below this median rate. The
    # incompatible Qwen2 runtime produces broadband codec noise whose samples
    # change sign almost randomly, so it crosses zero in roughly half of all
    # adjacent sample pairs.
    zero_crossings = np.mean(
        np.signbit(active_frames[:, 1:]) != np.signbit(active_frames[:, :-1]),
        axis=1,
    )
    if float(np.median(zero_crossings)) > _MAX_SPEECH_ZERO_CROSSING_RATE:
        raise RuntimeError(
            "CosyVoice 3 produced unintelligible codec noise. The generation was discarded."
        )


def _qwen2_compat_path() -> Path:
    """Return the private Transformers 4.40.1 Qwen2 implementation path."""
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(frozen_root) / "CosyVoice" / "compat" / "modeling_qwen2_440.py"
    return (
        Path(__file__).resolve().parent.parent
        / "vendors"
        / "cosyvoice_compat"
        / "modeling_qwen2_440.py"
    )


def _install_qwen2_compat() -> type:
    """Install the checkpoint-compatible Qwen2 class for CosyVoice only.

    Fun-CosyVoice3-0.5B-2512 records Transformers 4.40.1 in its config. Newer
    Qwen2 cache and rotary-position APIs accumulate enough autoregressive drift
    to produce repeated syllables or noise. Voicebox also needs newer
    Transformers for other engines, so the published 4.40.1 implementation is
    loaded under a private module name and injected only into CosyVoice's
    ``Qwen2Encoder`` constructor.
    """
    from transformers.cache_utils import DynamicCache

    if not hasattr(DynamicCache, "get_usable_length"):
        # Transformers 4.40 asks an unbounded dynamic cache how much history is
        # usable. Newer releases renamed the operation to ``get_seq_length``.
        def get_usable_length(self, _new_sequence_length, layer_idx=0):
            return self.get_seq_length(layer_idx)

        DynamicCache.get_usable_length = get_usable_length

    compat_module = sys.modules.get(_QWEN2_COMPAT_MODULE)
    if compat_module is None:
        compat_path = _qwen2_compat_path()
        if not compat_path.is_file():
            raise RuntimeError(f"CosyVoice Qwen2 compatibility source is missing: {compat_path}")
        spec = importlib.util.spec_from_file_location(_QWEN2_COMPAT_MODULE, compat_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load CosyVoice Qwen2 compatibility source: {compat_path}")
        compat_module = importlib.util.module_from_spec(spec)
        sys.modules[_QWEN2_COMPAT_MODULE] = compat_module
        spec.loader.exec_module(compat_module)

    import cosyvoice.llm.llm as cosyvoice_llm

    cosyvoice_llm.Qwen2ForCausalLM = compat_module.Qwen2ForCausalLM
    return compat_module.Qwen2ForCausalLM


def _load_audio_with_soundfile(
    uri,
    frame_offset: int = 0,
    num_frames: int = -1,
    normalize: bool = True,
    channels_first: bool = True,
    **_unused,
):
    """TorchCodec-free replacement for CosyVoice's reference-audio loader."""
    import soundfile as sf
    import torch

    del normalize  # SoundFile returns float32 just like TorchCodec's decoder.
    frames = -1 if num_frames == -1 else num_frames
    samples, sample_rate = sf.read(
        uri,
        dtype="float32",
        always_2d=True,
        start=frame_offset,
        frames=frames,
    )
    waveform = torch.from_numpy(samples.T if channels_first else samples)
    return waveform, sample_rate


def _vendor_root() -> Path:
    """Return the audited CosyVoice source location in dev or frozen builds."""
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(frozen_root) / "CosyVoice"
    return Path(__file__).resolve().parent.parent / "vendors" / "CosyVoice"


def _prepare_imports() -> None:
    """Expose bundled source and avoid CosyVoice's training-only import graph.

    ``cosyvoice3.yaml`` contains HyperPyYAML references to dataset pipeline
    functions. The inference constructor never executes them, but HyperPyYAML
    still resolves every symbol eagerly. Importing the real pipeline would pull
    training-only dependencies such as pyarrow and pyworld into Voicebox.
    """
    root = _vendor_root()
    matcha_root = root / "third_party" / "Matcha-TTS"
    if not root.is_dir() or not matcha_root.is_dir():
        raise RuntimeError(
            "CosyVoice 3 runtime source is missing. Run `just setup-python` to install the pinned CosyVoice runtime."
        )

    for source_root in (str(root), str(matcha_root)):
        if source_root not in sys.path:
            sys.path.insert(0, source_root)

    # Torchaudio 2.9+ routes ``torchaudio.load`` through TorchCodec. CosyVoice
    # calls that API only to read a profile's reference WAV, while Voicebox
    # already ships SoundFile for reliable cross-platform audio I/O. Patch it
    # before importing upstream modules so cloning works without TorchCodec.
    import torchaudio

    if not getattr(torchaudio.load, "_voicebox_soundfile_loader", False):
        _load_audio_with_soundfile._voicebox_soundfile_loader = True
        torchaudio.load = _load_audio_with_soundfile

    def _noop(*_args, **_kwargs):
        return None

    dataset = types.ModuleType("cosyvoice.dataset")
    dataset.__path__ = []  # type: ignore[attr-defined]
    processor = types.ModuleType("cosyvoice.dataset.processor")
    for name in (
        "parquet_opener",
        "tokenize",
        "filter",
        "resample",
        "truncate",
        "compute_fbank",
        "compute_whisper_fbank",
        "compute_f0",
        "parse_embedding",
        "shuffle",
        "sort",
        "batch",
        "padding",
    ):
        setattr(processor, name, _noop)

    # Do not replace an already-imported real module: the backend may be
    # reloaded by Uvicorn during development.
    sys.modules.setdefault("cosyvoice.dataset", dataset)
    sys.modules.setdefault("cosyvoice.dataset.processor", processor)
    # HyperPyYAML resolves names through parent package attributes, so cached
    # module entries alone are insufficient for its eager dataset references.
    cosyvoice = __import__("cosyvoice")
    cosyvoice.dataset = sys.modules["cosyvoice.dataset"]  # type: ignore[attr-defined]
    cosyvoice.dataset.processor = sys.modules["cosyvoice.dataset.processor"]  # type: ignore[attr-defined]

    # Matcha's package initializer imports its training helpers, even though
    # CosyVoice inference only needs ``matcha.utils.pylogger``.  Present the
    # package without executing that initializer, then give pylogger its tiny
    # rank-zero decorator dependency. This avoids shipping PyTorch Lightning
    # and its training-only dependency graph in the desktop TTS runtime.
    matcha_utils = types.ModuleType("matcha.utils")
    matcha_utils.__path__ = [str(matcha_root / "matcha" / "utils")]  # type: ignore[attr-defined]
    sys.modules.setdefault("matcha.utils", matcha_utils)
    # HyperPyYAML resolves dotted names through attributes on the parent
    # package (rather than looking only in sys.modules), so expose the shim on
    # ``matcha`` as well.
    matcha = __import__("matcha")
    matcha.utils = sys.modules["matcha.utils"]  # type: ignore[attr-defined]

    try:
        __import__("lightning.pytorch.utilities")
    except ModuleNotFoundError:
        lightning = types.ModuleType("lightning")
        lightning_pytorch = types.ModuleType("lightning.pytorch")
        lightning_utilities = types.ModuleType("lightning.pytorch.utilities")
        lightning_utilities.rank_zero_only = lambda fn: fn
        lightning.pytorch = lightning_pytorch  # type: ignore[attr-defined]
        lightning_pytorch.utilities = lightning_utilities  # type: ignore[attr-defined]
        sys.modules.setdefault("lightning", lightning)
        sys.modules.setdefault("lightning.pytorch", lightning_pytorch)
        sys.modules.setdefault("lightning.pytorch.utilities", lightning_utilities)

    from cosyvoice.llm.llm import CosyVoice3LM

    _install_qwen2_compat()

    if not getattr(CosyVoice3LM.sampling_ids, "_voicebox_stop_token_guard", False):
        def _sampling_ids_with_stop_token_guard(
            self,
            weighted_scores,
            decoded_tokens,
            sampling,
            ignore_eos=True,
        ):
            # CosyVoice 3 reserves 200 control/stop tokens, whereas the
            # inherited CosyVoice 2 helper masks only ``speech_token_size``.
            # The RL checkpoint can otherwise choose a different stop token on
            # its first decoding step and return a 0.04-second clip. During the
            # model's minimum-length window none of those control tokens may
            # terminate synthesis.
            if ignore_eos:
                weighted_scores[self.stop_token_ids] = -float("inf")
            return self.sampling(weighted_scores, decoded_tokens, sampling)

        _sampling_ids_with_stop_token_guard._voicebox_stop_token_guard = True
        CosyVoice3LM.sampling_ids = _sampling_ids_with_stop_token_guard


class CosyVoiceTTSBackend:
    """CosyVoice 3 0.5B backend; RL is the default reading-quality variant."""

    _import_lock: ClassVar[threading.Lock] = threading.Lock()
    _imports_prepared: ClassVar[bool] = False

    def __init__(self) -> None:
        self.model = None
        self.model_size = "rl"
        self._current_model_size: str | None = None
        self._device: str | None = None
        self._rl_model_view: tempfile.TemporaryDirectory[str] | None = None
        self._model_load_lock = asyncio.Lock()
        self._worker_lock = asyncio.Lock()
        self._worker = None
        self._worker_requests = None
        self._worker_events = None
        self._worker_model_size: str | None = None
        self._phase_callback = None

    def _get_device(self) -> str:
        if getattr(self, "_voicebox_force_cpu", False):
            return "cpu"
        return get_torch_device(allow_mps=True)

    def _get_model_path(self, model_size: str = "rl") -> str:
        try:
            return COSYVOICE_HF_REPOS[model_size]
        except KeyError as exc:
            raise ValueError(f"Unknown CosyVoice 3 model variant: {model_size}") from exc

    def _is_model_cached(self, model_size: str | None = None) -> bool:
        variant = model_size or self.model_size
        required_files = [*_REQUIRED_FILES, "llm.rl.pt"] if variant == "rl" else [*_REQUIRED_FILES, "llm.pt"]
        return is_model_cached(self._get_model_path(variant), required_files=required_files)

    def is_loaded(self) -> bool:
        return self.model is not None or bool(self._worker and self._worker.is_alive())

    async def load_model(self, model_size: str = "rl") -> None:
        """Download and load the requested CosyVoice 3 variant once."""
        if model_size not in COSYVOICE_HF_REPOS:
            raise ValueError(f"Unknown CosyVoice 3 model variant: {model_size}")
        if self.model is not None and self._current_model_size == model_size:
            return

        async with self._model_load_lock:
            if self.model is not None and self._current_model_size == model_size:
                return
            if self.model is not None:
                self.unload_model()
            # The parent must not construct the upstream model: CosyVoice's
            # internal unbounded join would then be impossible to stop.  The
            # child is created lazily at first inference so a failed/expired
            # segment can be killed together with every upstream thread.
            self.model_size = model_size
            self._current_model_size = model_size

    def _load_model_sync(self, model_size: str) -> None:
        repo_id = self._get_model_path(model_size)
        with model_load_progress(f"cosyvoice3-0.5b-{model_size}", self._is_model_cached(model_size)):
            with self._import_lock:
                if not self._imports_prepared:
                    _prepare_imports()
                    type(self)._imports_prepared = True

            # Download explicitly through huggingface_hub. The upstream
            # constructor otherwise chooses ModelScope, which prevents
            # Voicebox's Hub progress tracking and offline-cache policy from
            # applying consistently across TTS engines.
            from huggingface_hub import snapshot_download

            # TensorRT and batch-tokenizer artifacts are optional upstream
            # accelerators that this backend never loads. The two LLM weights
            # are alternatives, so fetch only the selected one. This avoids
            # downloading roughly 4 GB of unused files per model selection.
            ignored_files = ["flow.decoder.estimator.fp32.onnx", "speech_tokenizer_v3.batch.onnx"]
            ignored_files.append("llm.pt" if model_size == "rl" else "llm.rl.pt")
            model_dir = Path(
                snapshot_download(
                    repo_id=repo_id,
                    token=None,
                    ignore_patterns=ignored_files,
                    local_files_only=self._is_model_cached(model_size),
                )
            )
            if model_size == "rl":
                model_dir = self._create_rl_model_view(model_dir)
            from cosyvoice.cli.cosyvoice import CosyVoice3

            self._device = self._get_device()
            logger.info("Loading CosyVoice 3 (%s) on %s", model_size, self._device)
            try:
                self.model = CosyVoice3(model_dir=str(model_dir), fp16=self._device == "cuda")
            except Exception:
                # A failed constructor can occur after the RL symlink view is
                # created (for example, an interrupted model download). Do not
                # retain that temporary view for the remainder of the server.
                if self._rl_model_view is not None:
                    self._rl_model_view.cleanup()
                    self._rl_model_view = None
                raise

        self.model_size = model_size
        self._current_model_size = model_size

    def _create_rl_model_view(self, source_dir: Path) -> Path:
        """Expose upstream's ``llm.rl.pt`` under CosyVoice's required name.

        The published constructor only opens ``llm.pt``.  Rather than renaming
        a file in the shared HuggingFace cache (which would corrupt the base
        variant and other processes), create a lightweight symlink view where
        ``llm.pt`` resolves to the RL checkpoint. No model weights are copied.
        """
        rl_checkpoint = source_dir / "llm.rl.pt"
        if not rl_checkpoint.is_file():
            raise FileNotFoundError(f"CosyVoice 3 RL checkpoint is missing: {rl_checkpoint}")

        self._rl_model_view = tempfile.TemporaryDirectory(prefix="voicebox-cosyvoice3-rl-")
        view_dir = Path(self._rl_model_view.name)
        for source_path in source_dir.rglob("*"):
            destination = view_dir / source_path.relative_to(source_dir)
            if source_path.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
            elif source_path.name != "llm.pt":
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(source_path.resolve(), destination)
        os.symlink(rl_checkpoint.resolve(), view_dir / "llm.pt")
        return view_dir

    def unload_model(self) -> None:
        self._terminate_worker()
        was_loaded = self.model is not None
        self.model = None
        self._current_model_size = None
        if self._rl_model_view is not None:
            self._rl_model_view.cleanup()
            self._rl_model_view = None
        device, self._device = self._device, None
        if not was_loaded:
            return
        gc.collect()
        if device:
            empty_device_cache(device)
        logger.info("CosyVoice 3 unloaded")

    def _terminate_worker(self) -> None:
        """Stop a worker and every internal upstream thread it owns."""
        worker, self._worker = self._worker, None
        requests, self._worker_requests = self._worker_requests, None
        events, self._worker_events = self._worker_events, None
        self._worker_model_size = None
        if worker is None:
            return
        try:
            if worker.is_alive() and requests is not None:
                requests.put_nowait(None)
                worker.join(timeout=0.5)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=2)
            if worker.is_alive():
                worker.kill()
                worker.join(timeout=1)
        finally:
            for channel in (requests, events):
                if channel is not None:
                    close = getattr(channel, "close", None)
                    if callable(close):
                        close()

    def _start_worker(self, model_size: str) -> None:
        if self._worker is not None and self._worker.is_alive() and self._worker_model_size == model_size:
            return
        self._terminate_worker()
        context = multiprocessing.get_context("spawn")
        self._worker_requests = context.Queue()
        self._worker_events = context.Queue()
        self._worker = context.Process(
            target=_cosyvoice_worker_main,
            args=(self._worker_requests, self._worker_events, model_size),
            name="voicebox-cosyvoice",
        )
        self._worker.daemon = True
        self._worker.start()
        self._worker_model_size = model_size

    def _generate_in_worker(
        self,
        text: str,
        voice_prompt: dict,
        seed: int | None,
        instruct: str | None,
    ) -> tuple[np.ndarray, int]:
        from .. import config

        self._start_worker(self._current_model_size or self.model_size)
        assert self._worker is not None
        assert self._worker_requests is not None
        assert self._worker_events is not None
        deadline = time.monotonic() + config.get_cosyvoice_segment_timeout_seconds()
        request_id = os.urandom(8).hex()
        phase = "preprocessing"
        submitted = False

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CosyVoiceSegmentTimeoutError(phase)
            try:
                event = self._worker_events.get(timeout=min(remaining, 0.25))
            except queue.Empty:
                if not self._worker.is_alive():
                    raise CosyVoiceWorkerError(phase) from None
                continue

            kind = event.get("kind")
            if kind == "startup_error":
                raise CosyVoiceWorkerError("preprocessing")
            if not submitted:
                if kind != "ready":
                    continue
                self._worker_requests.put(
                    {
                        "id": request_id,
                        "text": text,
                        "voice_prompt": voice_prompt,
                        "seed": seed,
                        "instruct": instruct,
                    }
                )
                submitted = True
                continue
            if event.get("id") != request_id:
                continue
            if kind == "phase":
                phase = event.get("phase") if event.get("phase") in _COSYVOICE_PHASES else phase
                if self._phase_callback is not None:
                    self._phase_callback(phase)
            elif kind == "result":
                return np.asarray(event["audio"], dtype=np.float32), int(event["sample_rate"])
            elif kind == "error":
                raise CosyVoiceWorkerError(event.get("phase") or phase)

    def set_phase_callback(self, callback) -> None:
        """Set a transient observer for safe worker phase transitions."""
        self._phase_callback = callback

    async def create_voice_prompt(
        self, audio_path: str, reference_text: str, use_cache: bool = True
    ) -> tuple[dict, bool]:
        # CosyVoice extracts the reference features immediately before each
        # synthesis. Store the canonical profile sample so cached Voicebox
        # profiles stay portable across model variants and application restarts.
        return {"ref_audio": str(audio_path), "ref_text": reference_text}, False

    async def combine_voice_prompts(self, audio_paths: list[str], reference_texts: list[str]) -> tuple[np.ndarray, str]:
        return await _combine_voice_prompts(audio_paths, reference_texts, sample_rate=_SAMPLE_RATE)

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: int | None = None,
        instruct: str | None = None,
    ) -> tuple[np.ndarray, int]:
        await self.load_model(self._current_model_size or self.model_size)
        ref_audio = voice_prompt.get("ref_audio")
        if not ref_audio or not Path(ref_audio).is_file():
            raise ValueError("CosyVoice 3 requires an existing reference audio sample for voice cloning.")

        # Unit tests and explicitly injected model instances stay in-process;
        # production starts with no parent model and therefore always uses the
        # terminable worker path below.
        if self.model is not None:
            return await asyncio.to_thread(
                self._generate_sync, text, voice_prompt, seed, instruct, None
            )

        async with self._worker_lock:
            try:
                return await asyncio.to_thread(
                    self._generate_in_worker, text, voice_prompt, seed, instruct
                )
            except BaseException:
                # Cancellation and deadline expiry must remove the worker,
                # otherwise its upstream join can survive and retain memory.
                self._terminate_worker()
                raise

    def _generate_sync(
        self,
        text: str,
        voice_prompt: dict,
        seed: int | None,
        instruct: str | None,
        phase_callback,
    ) -> tuple[np.ndarray, int]:
        """Run one upstream inference while exposing only safe phase names."""
        import torch

        if self.model is None:
            raise RuntimeError("CosyVoice model is not loaded")
        ref_audio = voice_prompt["ref_audio"]
        if phase_callback is not None:
            phase_callback("preprocessing")
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        upstream = getattr(self.model, "model", None)
        restores = []

        def instrument(owner, attribute: str, phase: str) -> None:
            if owner is None or not hasattr(owner, attribute) or phase_callback is None:
                return
            original = getattr(owner, attribute)

            def wrapped(*args, **kwargs):
                phase_callback(phase)
                return original(*args, **kwargs)

            setattr(owner, attribute, wrapped)
            restores.append((owner, attribute, original))

        # The upstream's LLM runs in its own thread, followed by Flow and the
        # HiFT vocoder.  Wrapping their call sites reveals the real stall point
        # without persisting prompt text, paths or exceptions.
        instrument(upstream, "llm_job", "llm_decoding")
        instrument(getattr(upstream, "flow", None), "inference", "flow")
        instrument(getattr(upstream, "hift", None), "inference", "vocoder")
        try:
            if instruct:
                outputs = self.model.inference_instruct2(
                    tts_text=text,
                    instruct_text=_format_instruct_prompt(instruct),
                    prompt_wav=ref_audio,
                    stream=False,
                )
            else:
                outputs = self.model.inference_zero_shot(
                    tts_text=text,
                    prompt_text=_format_reference_prompt(voice_prompt.get("ref_text", "")),
                    prompt_wav=ref_audio,
                    stream=False,
                )
            chunks = [output["tts_speech"] for output in outputs]
            if not chunks:
                raise RuntimeError("CosyVoice 3 returned no audio.")
            audio = torch.cat(chunks, dim=-1).squeeze().detach().cpu().numpy().astype(np.float32)
            _validate_generated_audio(audio, self.model.sample_rate, text)
            return audio, self.model.sample_rate
        finally:
            for owner, attribute, original in reversed(restores):
                setattr(owner, attribute, original)


assert isinstance(CosyVoiceTTSBackend(), TTSBackend)
