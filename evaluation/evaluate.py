"""Portable, manifest-based evaluation of exact BSWS listening stimuli.

The numerical kernels are preserved in legacy_metrics.py and utmos_metric.py.
This release wrapper contains no audio URLs, private storage IDs, checkpoints,
participant data, or training/checkpoint-selection automation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from pathlib import Path

from legacy_metrics import METRICS, MetricEvaluator
from utmos_metric import UTMOSScorer

METRIC_DIRECTIONS = {
    "mcd": -1, "plcc": 1, "ssim": 1, "mstft": -1, "pesq": 1,
    "periodicity_error": -1, "pitch_error": -1, "voicing_f1": 1, "utmos": 1,
}
SEED = 20260923


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_manifest(path, audio_root):
    """Validate mono 24-kHz WAVs, duplicate keys, and explicit GT pairing."""
    import wave
    root = Path(audio_root).resolve()
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"utterance_id", "model_key", "path"} <= set(reader.fieldnames or []):
            raise ValueError("Manifest requires utterance_id, model_key, path columns")
        rows = list(reader)
    seen = set()
    for row in rows:
        key = (row["utterance_id"], row["model_key"])
        if not all(key) or key in seen:
            raise ValueError(f"Empty or duplicate utterance/model key: {key}")
        seen.add(key)
        relative = Path(row["path"])
        if relative.is_absolute():
            raise ValueError("Use relative audio paths in the public manifest")
        audio = (root / relative).resolve()
        if not audio.is_relative_to(root):
            raise ValueError("Manifest path escapes --audio-root")
        if not audio.is_file():
            raise FileNotFoundError(f"Missing manifest audio: {relative}")
        with wave.open(str(audio), "rb") as wav:
            if (wav.getnchannels(), wav.getframerate(), wav.getsampwidth()) != (1, 24000, 2):
                raise ValueError(f"Expected mono 24-kHz PCM16 WAV: {relative}")
            if wav.getnframes() < 1:
                raise ValueError(f"Empty audio: {relative}")
        actual = sha256(audio)
        if row.get("sha256") and row["sha256"].lower() != actual:
            raise ValueError(f"Audio SHA-256 mismatch: {relative}")
        row.update(resolved_path=audio, sha256=actual)
    references = {row["utterance_id"] for row in rows if row["model_key"] == "GT"}
    for row in rows:
        if row["utterance_id"] not in references:
            raise ValueError(f"No GT row for {row['utterance_id']}")
    if not rows or not any(row["model_key"] != "GT" for row in rows):
        raise ValueError("Manifest must contain GT and at least one generated condition")
    return rows


def paired_summary(rows_a, rows_b, metric, expected_ids=None, resamples=20000):
    """Same finite utterances in both models; positive benefit favors A."""
    import numpy as np
    a = {row["utterance_id"]: finite(row.get(metric)) for row in rows_a}
    b = {row["utterance_id"]: finite(row.get(metric)) for row in rows_b}
    if len(a) != len(rows_a) or len(b) != len(rows_b):
        raise ValueError("Duplicate utterances in model comparison")
    union = sorted(set(a) | set(b) | set(expected_ids or []))
    included = [uid for uid in union if a.get(uid) is not None and b.get(uid) is not None]
    result = dict(metric=metric, n_common=len(included), included_ids=included,
                  excluded_ids=[uid for uid in union if uid not in included],
                  mean_a=None, mean_b=None, raw_a_minus_b=None, benefit_a=None,
                  ci95_low=None, ci95_high=None)
    if not included:
        return result
    av, bv = np.array([a[uid] for uid in included]), np.array([b[uid] for uid in included])
    differences = (av - bv) * METRIC_DIRECTIONS[metric]
    result.update(mean_a=float(av.mean()), mean_b=float(bv.mean()),
                  raw_a_minus_b=float((av - bv).mean()), benefit_a=float(differences.mean()))
    if len(included) >= 2:
        rng = np.random.default_rng(SEED)
        samples = differences[rng.integers(0, len(included), (resamples, len(included)))].mean(axis=1)
        lower, upper = np.quantile(samples, [.025, .975])
        result.update(ci95_low=float(lower), ci95_high=float(upper))
    return result


def save_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New output directory; existing directories are rejected")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--metrics", nargs="+", choices=tuple(METRIC_DIRECTIONS), default=list(METRICS))
    parser.add_argument("--compare", nargs=2, action="append", metavar=("MODEL_A", "MODEL_B"), default=[])
    parser.add_argument("--validate-only", action="store_true", help="Validate manifest/audio without metric imports")
    args = parser.parse_args(argv)
    rows = read_manifest(args.manifest, args.audio_root)
    keys = {row["model_key"] for row in rows}
    for pair in args.compare:
        if set(pair) - (keys - {"GT"}) or pair[0] == pair[1]:
            parser.error("Comparisons need two distinct generated model keys present in the manifest")
    if args.validate_only:
        print(json.dumps({"valid_audio_files": len(rows), "model_keys": sorted(keys)}))
        return 0
    if args.output.exists():
        parser.error("--output must not already exist; previous results are never overwritten")
    selected = list(dict.fromkeys(args.metrics))
    reference_metrics = [metric for metric in selected if metric in METRICS]
    evaluator = MetricEvaluator(device=args.device, crepe_batch_size=512)
    preflight = evaluator.preflight(reference_metrics)
    if not preflight["ok"]:
        print(json.dumps(preflight, indent=2), file=sys.stderr)
        return 2
    scorer = UTMOSScorer(device=args.device) if "utmos" in selected else None
    if scorer:
        scorer.prepare()  # May download pinned public code/weights; no silent fallback.
    import numpy as np
    args.output.mkdir(parents=True)
    references = {row["utterance_id"]: row for row in rows if row["model_key"] == "GT"}
    results, diagnostics = [], []
    metric_fields = ["utterance_id", "model_key", "reference_sha256", "generated_sha256", *selected]
    for row in rows:
        if row["model_key"] == "GT":
            continue
        ref = references[row["utterance_id"]]
        pair_seed = int.from_bytes(hashlib.sha256(f"{SEED}:{ref['sha256']}:{row['sha256']}".encode()).digest()[:4], "big")
        np.random.seed(pair_seed)  # Once per pair; ref then generated CREPE draws.
        measured = evaluator.evaluate_pair(ref["resolved_path"], row["resolved_path"], reference_metrics)
        if scorer:
            try:
                measured["utmos"] = scorer.score(row["resolved_path"])
            except Exception as exc:
                measured["utmos"] = None
                measured["errors"]["utmos"] = f"{type(exc).__name__}: {exc}"
        result = dict(utterance_id=row["utterance_id"], model_key=row["model_key"],
                      reference_sha256=ref["sha256"], generated_sha256=row["sha256"],
                      **{metric: finite(measured.get(metric)) for metric in selected})
        results.append(result)
        for metric, error in measured["errors"].items():
            diagnostics.append(dict(utterance_id=row["utterance_id"], model_key=row["model_key"], metric=metric, error=error))
        save_csv(args.output / "per_utterance_metrics.csv", results, metric_fields)
        save_csv(args.output / "diagnostics.csv", diagnostics, ["utterance_id", "model_key", "metric", "error"])
        print(f"{len(results)}: {row['utterance_id']} / {row['model_key']}", flush=True)
    comparisons = []
    for model_a, model_b in args.compare:
        for metric in selected:
            comparisons.append(dict(model_a=model_a, model_b=model_b,
                **paired_summary([row for row in results if row["model_key"] == model_a],
                                 [row for row in results if row["model_key"] == model_b],
                                 metric, expected_ids=references)))
    save_json(args.output / "paired_comparisons.json", comparisons)
    utmos = scorer.metadata if scorer else None
    if utmos:
        utmos.pop("weights_local_path", None)  # Do not publish cache paths.
    packages = {}
    for package in ("numpy", "scipy", "librosa", "soundfile", "pysptk", "fastdtw", "pesq", "torch", "torchaudio", "torchcrepe", "auraloss", "scikit-image", "scikit-learn"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    save_json(args.output / "provenance.json", dict(
        python=platform.python_version(), packages=packages, preflight=preflight,
        manifest_sha256=sha256(args.manifest), metrics=selected, seed=SEED,
        crepe_batch_size=512, utmos=utmos,
        code_sha256={name: sha256(Path(__file__).with_name(name)) for name in ("evaluate.py", "legacy_metrics.py", "utmos_metric.py")},
        comparison_scope="Paired finite utterances; exploratory percentile bootstrap (20000 resamples). Not listener/training-seed uncertainty.",
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
