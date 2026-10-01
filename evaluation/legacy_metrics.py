"""The eight reference-based metrics from ``Untitledtest (5).ipynb``.

This module preserves the metric definitions in zero-based cell 20 and the
metric-specific loading/preprocessing in cells 23 and 24. It does not execute
notebook cells, generate audio, trim the listening stimuli globally, use the
notebook's unrelated Benford preprocessing, or calculate FAD.

Important legacy conventions are intentional: MCD alone receives peak-normalized
waveforms; PLCC can return its original 0.0 fallback; M-STFT receives reference
first and generated audio second; pitch error is natural-log-F0 RMSE; and the
voicing-F1 validity test requires both classes in each waveform. Non-finite
results become NaN with an explicit diagnostic (the original aggregation also
excluded every non-finite result). All third-party imports are lazy.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import math
from pathlib import Path
from typing import Iterable


METRICS = (
    "mcd", "plcc", "ssim", "mstft", "pesq",
    "periodicity_error", "pitch_error", "voicing_f1",
)
PERIODICITY_METRICS = ("periodicity_error", "pitch_error", "voicing_f1")

_DEPENDENCIES = {
    "mcd": ("numpy", "librosa", "pysptk", "fastdtw", "scipy.spatial.distance"),
    "plcc": ("numpy", "librosa", "fastdtw", "scipy.spatial.distance", "scipy.stats"),
    "ssim": ("numpy", "librosa", "skimage.metrics"),
    "mstft": ("numpy", "librosa", "torch", "auraloss"),
    "pesq": ("numpy", "librosa", "pesq"),
    "periodicity_error": ("numpy", "librosa", "torch", "torchcrepe", "sklearn.metrics"),
    "pitch_error": ("numpy", "librosa", "torch", "torchcrepe", "sklearn.metrics"),
    "voicing_f1": ("numpy", "librosa", "torch", "torchcrepe", "sklearn.metrics"),
}
_DISTRIBUTIONS = {"skimage": "scikit-image", "sklearn": "scikit-learn"}
_REQUIRED_API = {
    "numpy": ("array", "append", "mean", "sqrt", "log"),
    "librosa": ("load", "effects.trim", "util.frame", "feature.melspectrogram", "power_to_db"),
    "pysptk": ("mcep",),
    "fastdtw": ("fastdtw",),
    "scipy.spatial.distance": ("euclidean",),
    "scipy.stats": ("pearsonr",),
    "skimage.metrics": ("structural_similarity",),
    "torch": ("device", "FloatTensor", "tensor", "no_grad", "cuda.is_available"),
    "auraloss": ("freq.MultiResolutionSTFTLoss",),
    "pesq": ("pesq",),
    "torchcrepe": ("predict",),
    "sklearn.metrics": ("f1_score",),
}


def _requested_metrics(metrics: Iterable[str] | str | None) -> tuple[str, ...]:
    if metrics is None:
        return METRICS
    if isinstance(metrics, str):
        metrics = (metrics,)
    result = tuple(dict.fromkeys(metrics))
    unknown = [metric for metric in result if metric not in METRICS]
    if unknown:
        raise ValueError(f"Unsupported metrics: {unknown}. Supported: {METRICS}")
    return result


class MetricEvaluator:
    """Evaluate exact on-disk listening WAVs using the legacy definitions.

    ``preflight()`` imports dependencies and reports import/device errors, but
    does not download CREPE weights or initialize an M-STFT loss. Missing
    dependencies never cause silent omission of a requested metric.

    ``evaluate_pair()`` returns requested metric values plus an ``errors`` dict.
    A failed/undefined metric is NaN. The sole intentional finite fallback is
    the notebook's PLCC=0.0 when no frame correlations survive; it is flagged in
    ``errors`` but left unchanged to preserve the original definition.

    CREPE's batch size is 512, matching the notebook. In torchcrepe 0.0.24,
    Viterbi postprocessing occurs separately within each batch: changing this
    value changes the decoded sequence, not only memory usage. Do not reduce it
    silently to handle memory pressure. Native pitch conversion also applies
    NumPy/SciPy triangular dither; seed control belongs to the caller and should
    seed once per pair, not reset the same seed separately for ref and gen.
    """

    def __init__(self, device: str = "auto", crepe_batch_size: int = 512):
        if not isinstance(device, str) or not device:
            raise ValueError("device must be a nonempty torch device name or 'auto'")
        if isinstance(crepe_batch_size, bool) or not isinstance(crepe_batch_size, int) or crepe_batch_size < 1:
            raise ValueError("crepe_batch_size must be a positive integer")
        self.requested_device = device
        self.crepe_batch_size = crepe_batch_size
        self._device = None
        self._modules = {}
        self._mstft = None

    def _module(self, name):
        if name not in self._modules:
            self._modules[name] = importlib.import_module(name)
        return self._modules[name]

    def _resolve_device(self):
        if self._device is None:
            torch = self._module("torch")
            chosen = self.requested_device
            if chosen == "auto":
                chosen = "cuda" if torch.cuda.is_available() else "cpu"
            device = torch.device(chosen)
            # Do not silently change an explicitly requested CUDA device to CPU.
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(f"Requested device {chosen!r}, but CUDA is unavailable")
            if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
                raise RuntimeError(f"Requested CUDA index {device.index} does not exist")
            self._device = str(device)
        return self._device

    def preflight(self, metrics: Iterable[str] | str | None = None) -> dict:
        """Return dependency import status, package versions, and device status."""
        selected = _requested_metrics(metrics)
        statuses = {}
        for name in dict.fromkeys(dep for metric in selected for dep in _DEPENDENCIES[metric]):
            try:
                module = self._module(name)
                for attribute_path in _REQUIRED_API[name]:
                    attribute = module
                    for part in attribute_path.split("."):
                        attribute = getattr(attribute, part)
                    if not callable(attribute):
                        raise TypeError(f"Required API {name}.{attribute_path} is not callable")
                root = name.split(".")[0]
                distribution = _DISTRIBUTIONS.get(root, root)
                try:
                    version = importlib.metadata.version(distribution)
                except importlib.metadata.PackageNotFoundError:
                    version = "unknown"
                statuses[name] = {"available": True, "version": version,
                                  "checked_api": list(_REQUIRED_API[name])}
            except Exception as exc:
                statuses[name] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
        device_error = None
        if any(metric == "mstft" or metric in PERIODICITY_METRICS for metric in selected):
            try:
                device = self._resolve_device()
            except Exception as exc:
                device = self.requested_device
                device_error = f"{type(exc).__name__}: {exc}"
        else:
            device = self._device or self.requested_device
        metric_status = {}
        for metric in selected:
            failures = {dep: statuses[dep]["error"] for dep in _DEPENDENCIES[metric]
                        if not statuses[dep]["available"]}
            if device_error and (metric == "mstft" or metric in PERIODICITY_METRICS):
                failures["device"] = device_error
            metric_status[metric] = {"available": not failures, "errors": failures}
        return {
            "ok": all(status["available"] for status in metric_status.values()),
            "device": device,
            "device_error": device_error,
            "crepe_batch_size": self.crepe_batch_size,
            "dependencies": statuses,
            "metrics": metric_status,
            "weights_checked": False,
        }

    def evaluate_pair(self, ref_path, gen_path, metrics: Iterable[str] | str | None = None) -> dict:
        selected = _requested_metrics(metrics)
        result = {metric: float("nan") for metric in selected}
        errors = {}
        result["errors"] = errors
        for path in (ref_path, gen_path):
            if not Path(path).is_file():
                message = f"FileNotFoundError: audio file not found: {path}"
                errors.update({metric: message for metric in selected})
                return result

        # Pair-local cache only: no persistent WAV/result cache can outlive file replacement.
        audio = {}

        def load_pair(sr):
            if sr not in audio:
                librosa = self._module("librosa")
                y1, _ = librosa.load(str(ref_path), sr=sr)
                y2, _ = librosa.load(str(gen_path), sr=sr)
                audio[sr] = (y1, y2)
            return audio[sr]

        for metric in selected:
            if metric in PERIODICITY_METRICS:
                continue
            try:
                if metric == "mcd":
                    np = self._module("numpy")
                    y1, y2 = load_pair(16000)
                    y1 = y1 / (np.max(np.abs(y1)) + 1e-9)
                    y2 = y2 / (np.max(np.abs(y2)) + 1e-9)
                    value = self.compute_mcd_sptk(y1, y2, sr=16000)
                elif metric == "plcc":
                    value, count = self.compute_plcc(*load_pair(16000), sr=16000)
                    if count == 0:
                        errors[metric] = "No finite frame correlations; preserved legacy PLCC fallback 0.0"
                elif metric == "ssim":
                    value = self.compute_ssim(str(ref_path), str(gen_path))
                elif metric == "mstft":
                    value = self.compute_mstft(*load_pair(24000), sr=24000)
                else:  # PESQ
                    value = self.compute_pesq(*load_pair(16000))
                value = float(value)
                if math.isfinite(value):
                    result[metric] = value
                else:
                    errors[metric] = f"Legacy metric returned non-finite value {value}; excluded as NaN"
            except Exception as exc:
                errors[metric] = f"{type(exc).__name__}: {exc}"

        requested_periodicity = [metric for metric in selected if metric in PERIODICITY_METRICS]
        if requested_periodicity:
            try:
                values, diagnostics = self.compute_periodicity(*load_pair(16000))
                for metric in requested_periodicity:
                    value = float(values[metric])
                    if math.isfinite(value):
                        result[metric] = value
                    else:
                        errors[metric] = diagnostics.get(metric, "Legacy metric returned a non-finite result")
            except Exception as exc:
                errors.update({metric: f"{type(exc).__name__}: {exc}" for metric in requested_periodicity})
        return result

    def extract_mcep(self, y, sr=16000, order=12, frame_length=256, frame_shift=64,
                     alpha=0.42, apply_trim=True):
        np = self._module("numpy")
        librosa = self._module("librosa")
        pysptk = self._module("pysptk")
        y = y - 0.97 * np.append(y[0], y[:-1])
        if apply_trim:
            y, _ = librosa.effects.trim(y, top_db=40)
        try:
            frames = librosa.util.frame(y, frame_length=frame_length, hop_length=frame_shift).T.copy()
        except Exception:
            return np.array([])
        win = np.hamming(frame_length)
        frames *= win
        mceps = []
        for frame in frames:
            try:
                mcep = pysptk.mcep(frame.astype(np.float64), order=order, alpha=alpha,
                                   min_det=0.0001, etype=1)
                mceps.append(mcep)
            except Exception:
                continue
        return np.array(mceps)

    def compute_mcd_sptk(self, y1, y2, sr=16000, order=24, alpha=0.42):
        np = self._module("numpy")
        fastdtw = self._module("fastdtw").fastdtw
        euclidean = self._module("scipy.spatial.distance").euclidean
        mcep1 = self.extract_mcep(y1, sr, order=order, alpha=alpha)
        mcep2 = self.extract_mcep(y2, sr, order=order, alpha=alpha)
        if len(mcep1) == 0 or len(mcep2) == 0:
            return float("inf")
        distance, path = fastdtw(mcep1, mcep2, dist=euclidean)
        if len(path) == 0:
            return float("inf")
        diff_sum = 0
        for i, j in path:
            diff = mcep1[i][1:] - mcep2[j][1:]
            diff_sum += np.sqrt(np.sum(diff ** 2))
        K = (10 / np.log(10)) * np.sqrt(2)
        return float(K * (diff_sum / len(path)))

    def compute_plcc(self, y1, y2, sr=16000, n_mels=80):
        np = self._module("numpy")
        librosa = self._module("librosa")
        fastdtw = self._module("fastdtw").fastdtw
        euclidean = self._module("scipy.spatial.distance").euclidean
        pearsonr = self._module("scipy.stats").pearsonr
        mel1 = librosa.feature.melspectrogram(y=y1, sr=sr, n_mels=n_mels)
        mel2 = librosa.feature.melspectrogram(y=y2, sr=sr, n_mels=n_mels)
        log_mel1 = librosa.power_to_db(mel1).T
        log_mel2 = librosa.power_to_db(mel2).T
        _, path = fastdtw(log_mel1, log_mel2, dist=euclidean)
        correlations = []
        for i, j in path:
            try:
                corr, _ = pearsonr(log_mel1[i], log_mel2[j])
                if not np.isnan(corr):
                    correlations.append(corr)
            except Exception:
                continue
        return (float(np.mean(correlations)) if correlations else 0.0), len(correlations)

    def wav_to_mel_image(self, path, sr=16000, n_mels=128):
        np = self._module("numpy")
        librosa = self._module("librosa")
        y, _ = librosa.load(path, sr=sr)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        S_dB = librosa.power_to_db(S, ref=np.max)
        S_norm = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min() + 1e-9)
        img_gray = (S_norm * 255).astype(np.uint8)
        return np.stack([img_gray] * 3, axis=-1)

    def compute_ssim(self, ref_path, gen_path, win_size=7):
        ssim_skimage = self._module("skimage.metrics").structural_similarity
        imgA = self.wav_to_mel_image(ref_path)
        imgB = self.wav_to_mel_image(gen_path)
        H = min(imgA.shape[0], imgB.shape[0])
        W = min(imgA.shape[1], imgB.shape[1])
        imgA = imgA[:H, :W]
        imgB = imgB[:H, :W]
        ssim_total = 0
        for i in range(3):
            ssim_val = ssim_skimage(imgA[:, :, i], imgB[:, :, i], win_size=win_size,
                                   data_range=imgB[:, :, i].max() - imgB[:, :, i].min())
            ssim_total += ssim_val
        return float(ssim_total / 3)

    def compute_mstft(self, y1, y2, sr=24000):
        torch = self._module("torch")
        device = self._resolve_device()
        if self._mstft is None:
            auraloss = self._module("auraloss")
            self._mstft = auraloss.freq.MultiResolutionSTFTLoss(
                fft_sizes=[1024, 2048, 512],
                hop_sizes=[120, 240, 50],
                win_lengths=[600, 1200, 240],
                scale="mel", n_bins=128,
                sample_rate=24000,
                perceptual_weighting=True,
                device=device,
            ).to(device)
        t1 = torch.FloatTensor(y1).unsqueeze(0).to(device)
        t2 = torch.FloatTensor(y2).unsqueeze(0).to(device)
        L = min(t1.size(-1), t2.size(-1))
        t1, t2 = t1[..., :L], t2[..., :L]
        with torch.no_grad():
            loss = self._mstft(t1.unsqueeze(0), t2.unsqueeze(0))
        return float(loss.item())

    def compute_pesq(self, y1_16k, y2_16k):
        pesq = self._module("pesq").pesq
        L = min(len(y1_16k), len(y2_16k))
        y1_16k, y2_16k = y1_16k[:L], y2_16k[:L]
        # evaluate_pair records exception details instead of silently returning NaN.
        return float(pesq(16000, y1_16k, y2_16k, "wb"))

    def compute_periodicity(self, y1_16k, y2_16k):
        np = self._module("numpy")
        torch = self._module("torch")
        torchcrepe = self._module("torchcrepe")
        f1_score = self._module("sklearn.metrics").f1_score
        device = self._resolve_device()

        def extract(y):
            t = torch.tensor(y).unsqueeze(0).float().to(device)
            pitch, periodicity = torchcrepe.predict(
                t, 16000, hop_length=80, fmin=50, fmax=550,
                model="full", return_periodicity=True,
                device=device, batch_size=self.crepe_batch_size,
            )
            return pitch.squeeze().cpu().numpy(), periodicity.squeeze().cpu().numpy()

        p1, per1 = extract(y1_16k)
        p2, per2 = extract(y2_16k)
        L = min(len(p1), len(p2))
        p1, per1, p2, per2 = p1[:L], per1[:L], p2[:L], per2[:L]
        voiced1 = per1 > 0.5
        voiced2 = per2 > 0.5
        periodicity_error = float(np.sqrt(np.mean((per1 - per2) ** 2)))
        diagnostics = {}
        both_voiced = voiced1 & voiced2
        if both_voiced.sum() > 0:
            log_p1 = np.log(p1[both_voiced] + 1e-9)
            log_p2 = np.log(p2[both_voiced] + 1e-9)
            pitch_error = float(np.sqrt(np.mean((log_p1 - log_p2) ** 2)))
        else:
            pitch_error = float("nan")
            diagnostics["pitch_error"] = "No jointly voiced frames under legacy periodicity > 0.5 rule"
        if len(np.unique(voiced1)) > 1 and len(np.unique(voiced2)) > 1:
            voicing_f1 = float(f1_score(voiced1, voiced2))
        else:
            voicing_f1 = float("nan")
            diagnostics["voicing_f1"] = "Legacy rule requires both voicing classes in each waveform"
        return {
            "periodicity_error": periodicity_error,
            "pitch_error": pitch_error,
            "voicing_f1": voicing_f1,
        }, diagnostics


def preflight(metrics: Iterable[str] | str | None = None, device: str = "auto",
              crepe_batch_size: int = 512) -> dict:
    """Convenience preflight without retaining an evaluator instance."""
    return MetricEvaluator(device=device, crepe_batch_size=crepe_batch_size).preflight(metrics)
