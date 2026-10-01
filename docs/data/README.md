# BSWS data documentation

## Release scope

The release contains the fixed 20-utterance listening stimuli, aggregated human scores, and automatic measurements on those same stimuli. It does not include participant-level ratings or data from the response-collection service. Automatic results here must not be mislabeled as the complete 4,837-utterance test-clean evaluation.

The subjective snapshot was verified on 2026-09-22. The matched objective reanalysis was finalized on 2026-09-23. Private session IDs and individual completion timestamps are omitted.

## Conditions

`GT` is reference speech. `Snake_MPD_MRD`, `Snake_MRD_only`, `LeakyReLU_MPD_MRD`, and `LeakyReLU_MRD_only` form the main comparison. `HiFiGAN_AMP` is a contextual reference system. Generator anti-aliasing is retained under both Snake and LeakyReLU. The public audio manifest reports author-identified checkpoint steps; a WAV hash identifies an audio file, not a checkpoint weight file.

## audio_manifest.csv

One row per WAV. `path` is relative to the website's `docs/` directory (or the audio ZIP root). `utterance_id` is U01–U20. `reference_id` is the LibriTTS corpus identifier, not a listening-test participant identifier. Sample rate is in hertz, duration in seconds, and `frames` counts audio samples per channel. All files are mono, 24 kHz, 16-bit PCM. `sha256` hashes the complete WAV bytes.

Both tasks use the same fixed 20 utterances. MOS includes all six conditions; CMOS main trials compare MPD+MRD with MRD-only separately for Snake and LeakyReLU. Files were copied into this release without recoding, normalization or trimming. Reference files are selected corpus speech; synthesized files are model reconstructions. LibriTTS was prepared by Heiga Zen with Google Speech and Google Brain team members. Source: https://www.openslr.org/60/ . License: https://creativecommons.org/licenses/by/4.0/ . Original paper: Heiga Zen et al., “LibriTTS: A Corpus Derived from LibriSpeech for Text-to-Speech,” 2019.

## listening_summary.csv and study_summary.json

For `mos`, each of 20 listeners rates 20 utterances for each of 6 conditions (2,400 main ratings). MOS ranges from 1 to 5, higher is better. `mos_paired_full_minus_mrd_only` is the within-listener difference.

For `cmos_full_minus_mrd_only`, each of 20 listeners completes 20 comparisons per activation (800 main ratings). Stored A/B scores are recoded so positive values favor MPD+MRD. CMOS ranges from −3 to +3. Within each activation, MPD+MRD appears as A 10 times and as B 10 times per listener.

Each result is an average of listener means. SD is the sample SD of those means. CI is mean ± t(0.975,19) × SD / sqrt(20). Intervals condition on the fixed utterances and selected checkpoints, rather than variation over training seeds or sampled utterances. Two listeners participated in both tasks, giving 38 distinct listeners. All 40 included non-preview sessions are complete and passed both instructed-response attention checks. Practice and checks do not contribute to the main quality scores.

## objective_paired.csv

One row per activation and metric. Means use the common valid utterances of the model pair; `n_common=20`. `raw_full_minus_only` is MPD+MRD minus MRD-only. `benefit_full` reverses this sign for lower-is-better metrics, so positive adjusted differences always favor retaining MPD. `ci95_low` and `ci95_high` describe this adjusted difference, not the individual model means. They are exploratory paired-utterance percentile-bootstrap intervals (20,000 resamples, seed 20260923), not listener-level or training-seed intervals.

Lower is better for MCD, M-STFT, periodicity error and natural-log pitch error. Higher is better for PLCC, SSIM, PESQ, voicing F1 and UTMOS. The first eight measures are reference-based; UTMOS22 strong is learned and non-reference. Means alone do not establish significance; in particular, the Snake pitch-error interval includes zero.

## objective_per_utterance.csv

Each row represents one condition and utterance, with a separate column for each metric. Only measurement fields and scientific identifiers are included. GT rows have UTMOS but no reference-based scores. See the evaluation code for metric definitions, input conventions and version requirements. Missing values are unavailable, not zero. This release excludes FAD and Lizard configurations.
