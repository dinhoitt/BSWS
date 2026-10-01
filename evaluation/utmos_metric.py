"""Pinned UTMOS22 strong inference for exact, unmodified listening stimuli.

This is the original UTMOS22 strong predictor, not UTMOSv2 or the full
UTMOS challenge ensemble.  The public ComVo implementation uses the same
SpeechMOS v1.2.0 loading path.  Dependencies and model weights are loaded
only when ``score`` is first called with a valid waveform.

No peak/RMS normalization, silence removal, output-score clipping,
duration cap, chunking, or cross-utterance padding is applied here.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import math
from pathlib import Path
from typing import Any


SPEECHMOS_COMMIT = "ed25eacbfa42b99156c36ebec67a733b5dbb9b79"
SPEECHMOS_REPOSITORY = f"tarepan/SpeechMOS:{SPEECHMOS_COMMIT}"
UTMOS_ENTRYPOINT = "utmos22_strong"
WEIGHTS_FILENAME = "utmos22_strong_step7459_v1.pt"
WEIGHTS_URL = (
    "https://github.com/tarepan/SpeechMOS/releases/download/v1.0.0/"
    + WEIGHTS_FILENAME
)


def _installed_version(distribution: str) -> str | None:
    """Read package metadata without importing its potentially heavy module."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


class UTMOSScorer:
    """Score one complete mono WAV at a time with a frozen UTMOS22 model.

    ``device='auto'`` selects CUDA if available, otherwise CPU.  Explicit
    device strings accepted by ``torch.device`` are also supported.

    Accessing ``metadata`` or ``provenance`` does not load the model or
    download any files.  After inference they include the actual cache
    location and, if present, a locally computed SHA-256 of the weights.
    The hash records the file used; it is not an independently published
    upstream checksum or a claim of authenticity verification.
    """

    def __init__(self, device: str = "auto") -> None:
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be 'auto' or a nonempty device string")
        self.requested_device = device.strip()
        self._device: str | None = None
        self._torch: Any = None
        self._model: Any = None
        self._weights_path: Path | None = None
        self._hash_key: tuple[str, int, int] | None = None
        self._weights_sha256: str | None = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            torch = importlib.import_module("torch")
            # Check the audio extension before downloading weights.  Its
            # build must match PyTorch (including CPU/CUDA variant).
            importlib.import_module("torchaudio")
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "UTMOS requires compatible torch and torchaudio builds. "
                "No substitute score was produced."
            ) from exc

        selected_device = self.requested_device
        if selected_device == "auto":
            selected_device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = str(torch.device(selected_device))
        self._torch = torch
        self._weights_path = (
            Path(torch.hub.get_dir()) / "checkpoints" / WEIGHTS_FILENAME
        )

        try:
            # Explicitly trust only this pinned public implementation.  Its
            # hub loader reads a tensor-only state_dict.  Do not globally
            # override torch.load or disable PyTorch's weights-only safety
            # behavior for torch >= 2.6 to accommodate unrelated checkpoints.
            model = torch.hub.load(
                SPEECHMOS_REPOSITORY,
                UTMOS_ENTRYPOINT,
                trust_repo=True,
                verbose=False,
            )
            model = model.float().eval().to(self._device)
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        except Exception as exc:
            raise RuntimeError(
                "Could not load the pinned SpeechMOS/UTMOS22 strong model. "
                "Check network/cache and torch/torchaudio compatibility. "
                "No fallback model or unsafe torch.load override was used."
            ) from exc
        self._model = model

    def score(self, path: str | Path) -> float:
        """Return one prediction for the complete mono audio file at ``path``.

        Original sample rate is passed to SpeechMOS, whose forward method
        performs 16-kHz resampling.  Each file is its own batch of size one,
        avoiding padding different-length utterances into a batch.  Invalid
        audio or inference failure raises an exception rather than returning
        a sentinel that could silently enter an aggregate.
        """
        audio_path = Path(path).expanduser().resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        try:
            np = importlib.import_module("numpy")
            sf = importlib.import_module("soundfile")
        except ImportError as exc:
            raise RuntimeError("UTMOS waveform loading requires numpy and soundfile") from exc

        try:
            waveform, sample_rate = sf.read(
                str(audio_path), dtype="float32", always_2d=True
            )
        except Exception as exc:
            raise ValueError(f"Cannot decode UTMOS input: {audio_path}") from exc
        if waveform.shape[1] != 1:
            raise ValueError(
                f"UTMOS expects the original mono stimulus, got "
                f"{waveform.shape[1]} channels: {audio_path}. "
                "No automatic channel mixing was applied."
            )
        if sample_rate <= 0 or waveform.shape[0] == 0:
            raise ValueError(f"Empty waveform or invalid sample rate: {audio_path}")
        if not np.isfinite(waveform).all():
            raise ValueError(f"Non-finite waveform samples: {audio_path}")

        self._load_model()
        tensor = self._torch.from_numpy(
            np.ascontiguousarray(waveform[:, 0])
        ).unsqueeze(0).to(self._device)
        try:
            with self._torch.inference_mode():
                prediction = self._model(tensor, sr=int(sample_rate))
            if prediction.numel() != 1:
                raise ValueError(f"Expected one UTMOS score, got {prediction.shape}")
            result = float(prediction.item())
        except Exception as exc:
            raise RuntimeError(
                f"UTMOS inference failed for {audio_path}. "
                "The waveform was not truncated, normalized, or chunked."
            ) from exc
        if not math.isfinite(result):
            raise ValueError(f"Non-finite UTMOS prediction: {audio_path}")
        return result

    def prepare(self) -> dict[str, Any]:
        """Load once and expose actual weight identity before any cache lookup."""
        self._load_model()
        return self.provenance(hash_weights=True)

    @property
    def metadata(self) -> dict[str, Any]:
        """Return provenance, including weight hash when weights are available."""
        return self.provenance(hash_weights=True)

    def provenance(self, *, hash_weights: bool = True) -> dict[str, Any]:
        """Describe the exact predictor and input policy without downloading.

        Pass ``hash_weights=False`` to skip reading the cached weight file.
        Hashes are memoized until the path, file size, or modification time
        changes.  Call after the first prediction for loaded-model provenance.
        """
        weights_path: str | None = None
        weights_size: int | None = None
        weights_hash: str | None = None
        if self._weights_path is not None and self._weights_path.is_file():
            resolved = self._weights_path.resolve()
            stat = resolved.stat()
            weights_path = str(resolved)
            weights_size = stat.st_size
            key = (weights_path, stat.st_size, stat.st_mtime_ns)
            if hash_weights:
                if self._hash_key != key:
                    digest = hashlib.sha256()
                    with resolved.open("rb") as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(block)
                    self._weights_sha256 = digest.hexdigest()
                    self._hash_key = key
                weights_hash = self._weights_sha256
        return {
            "metric": "UTMOS22 strong",
            "variant": "main-track strong learner without phoneme encoder",
            "direction": "higher_is_better",
            "implementation": "tarepan/SpeechMOS",
            "implementation_release": "v1.2.0",
            "implementation_commit": SPEECHMOS_COMMIT,
            "torch_hub_repository": SPEECHMOS_REPOSITORY,
            "torch_hub_entrypoint": UTMOS_ENTRYPOINT,
            "weights_filename": WEIGHTS_FILENAME,
            "weights_url": WEIGHTS_URL,
            "weights_local_path": weights_path,
            "weights_size_bytes": weights_size,
            "weights_sha256": weights_hash,
            "model_loaded": self._model is not None,
            "requested_device": self.requested_device,
            "actual_device": self._device,
            "input_dtype": "float32",
            "input_channels": "mono_required",
            "batch_size": 1,
            "input_extent": "complete_supplied_waveform",
            "model_sample_rate_hz": 16000,
            "resampling": "SpeechMOS internal torchaudio.functional.resample",
            "additional_normalization": "none",
            "silence_trimming": False,
            "chunking": False,
            "duration_cap_seconds": None,
            "output_score_clipping": False,
            "versions": {
                name: _installed_version(name)
                for name in ("torch", "torchaudio", "numpy", "soundfile")
            },
            "code_license": "MIT",
            "sources": {
                "utmos_official": "https://github.com/sarulab-speech/UTMOS22",
                "speechmos": "https://github.com/tarepan/SpeechMOS/tree/v1.2.0",
                "comvo_loading_path": (
                    "https://github.com/hs-oh-prml/ComVo/blob/main/exp/experiment.py"
                ),
            },
        }
