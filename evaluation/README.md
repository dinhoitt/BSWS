# BSWS: matched-stimulus objective evaluation

Code accompanying **Better Scores, Worse Speech: An Objective-Subjective Ranking
Reversal in BigVGAN-Style Discriminator Ablations**.

This is a minimal public release for evaluating the exact WAVs used in the
listening study. It preserves the eight reference-based metric implementations
from the research evaluator and optionally adds pinned UTMOS22 strong inference.
It does not retrain models, choose checkpoints, collect participant responses,
or reproduce the original full-test synthesis stage. No FAD is computed.

## Environment

Use a separate **Python 3.12** environment. Install PyTorch and torchaudio with
matching build variants first, then the requirements:

```sh
python -m venv .venv
# Activate .venv using your shell's normal activation command.
python -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

The CUDA 12.1 command is for a compatible NVIDIA environment. A CPU installation
can instead use the PyTorch CPU wheel index and `--device cpu`; it may be slow.
Metric extensions such as `pysptk`, `fastdtw`, and `pesq` can require a C/C++
compiler when a wheel is unavailable. Package licenses apply separately; in
particular, review the PESQ implementation's terms for your intended use.

The pins describe the archived local matched-stimulus reanalysis environment,
not necessarily the environment used for every historical full-test result.
Cross-platform floating-point differences are possible. Python 3.13 is not
supported by this requirements file; do not silently substitute its NumPy lane.

## Inputs and usage

The manifest is a CSV with one row per mono 24-kHz PCM16 WAV:

```csv
utterance_id,model_key,path,sha256
U01,GT,audio/GT/U01.wav,
U01,Snake_MPD_MRD,audio/Snake_MPD_MRD/U01.wav,
U01,Snake_MRD_only,audio/Snake_MRD_only/U01.wav,
```

`path` is relative to `--audio-root`; use forward slashes. `sha256` is optional,
but when supplied it is verified before evaluation. Every generated stimulus
must have one GT row with the same utterance ID. Duplicate keys, path traversal,
missing files, and incompatible audio formats are rejected. Supply the exact
published listening WAVs, not separately re-synthesized alternatives.

```sh
python evaluate.py --manifest manifest.csv --audio-root . --output results --validate-only
python evaluate.py --manifest manifest.csv --audio-root . --output results --device cuda --compare Snake_MPD_MRD Snake_MRD_only --compare LeakyReLU_MPD_MRD LeakyReLU_MRD_only
```

For a BSWS repository checkout with this directory at `evaluation/`, run from
the repository root using its public manifest and WAVs:

```sh
python evaluation/evaluate.py --manifest docs/data/audio_manifest.csv --audio-root docs --output objective-results --device cuda --compare Snake_MPD_MRD Snake_MRD_only --compare LeakyReLU_MPD_MRD LeakyReLU_MRD_only
```

The second command computes all eight reference-based metrics. To add UTMOS,
explicitly list the requested metrics:

```sh
python evaluate.py --manifest manifest.csv --audio-root . --output results-with-utmos --device cuda --metrics mcd plcc ssim mstft pesq periodicity_error pitch_error voicing_f1 utmos --compare Snake_MPD_MRD Snake_MRD_only --compare LeakyReLU_MPD_MRD LeakyReLU_MRD_only
```

The output directory must not already exist. Results are saved after each
completed audio pair, but this minimal wrapper does not resume a partial run.
Existing data are never overwritten. Validate-only creates no output.

Outputs are `per_utterance_metrics.csv`, `diagnostics.csv`,
`paired_comparisons.json`, and `provenance.json`. Invalid measurements are blank
CSV cells / JSON null, not zero. The legacy PLCC=0 fallback is deliberately
retained and explicitly flagged. Diagnostics may contain paths from your local
run; inspect them before publishing your outputs. GT rows are references, not
scored output rows; optional UTMOS is evaluated for generated conditions only.

## Metric conventions that must not be changed silently

- MCD alone peak-normalizes the input, then uses its own trim and DTW alignment.
- PLCC aligns log-mel frames with DTW; it is not waveform correlation.
- SSIM compares independently scaled mel images over their common extent.
- M-STFT uses mel-scaled, perceptually weighted `auraloss` with the reference
  passed first, preserving the original calling convention.
- PESQ is wideband at 16 kHz over the common waveform extent.
- CREPE uses the full model, 16 kHz, 80-sample hop, 50--550 Hz, and **batch 512**.
  Batch size affects torchcrepe 0.0.24 Viterbi decoding, not only memory use.
  Periodicity threshold is >0.5; pitch error is natural-log-F0 RMSE over jointly
  voiced frames. Voicing F1 requires both voicing classes in each waveform.
- The wrapper seeds NumPy once per reference/generated pair from SHA-256 of
  `20260923:reference_audio_hash:generated_audio_hash`, before reference then
  generated CREPE inference. This is the matched-stimulus reanalysis policy,
  not a claim that all historical evaluations controlled CREPE dither.
- UTMOS is a learned **non-reference** quality predictor. SpeechMOS commit
  `ed25eacbfa42b99156c36ebec67a733b5dbb9b79` supplies `utmos22_strong`.
  Full mono WAVs are scored individually in float32, without added normalization,
  trimming, chunking, score clipping, or cross-utterance padding. Its loader
  resamples internally to 16 kHz. The first use downloads and executes this
  pinned public Torch Hub implementation and downloads weights; review it
  before running. No weights are redistributed in this bundle.

Paired comparisons use the same finite utterances for both models, separately
for each metric. `benefit_a` is direction-adjusted so positive values favor the
first model named with `--compare`. Intervals are exploratory paired
utterance-level percentile bootstrap intervals (20,000 resamples; seed
20260923), conditional on this stimulus set and checkpoint choice. They are
not listener-level or training-seed uncertainty. The metrics are not eight
independent perceptual judgments.

## Scope and provenance

The main study checkpoint rule was minimum recorded generator total loss among
saved checkpoints **inside epoch 500**. This release only evaluates supplied
audio; it does not attempt to reconstruct that selection. Historical audit
scripts and the author-specific Colab notebook are intentionally not included:
they contain private storage mappings and unrelated exploratory logic.

`legacy_metrics.py` and `utmos_metric.py` are copied without numerical changes
from the audited research implementation. `evaluate.py` is a portable release
wrapper. Tests cover common-valid pairing, direction conventions, deterministic
bootstrap, exact metric defaults, manifest integrity, and privacy boundaries;
they do not substitute for reproducing the full evaluation.

```sh
python -m unittest discover -s tests -v
```

See [NOTICE.md](NOTICE.md) for dependencies and attribution. No participant-level
responses, personal identifiers, credentials, trained vocoder checkpoints,
Torch Hub caches, or private cloud storage IDs are included here.
