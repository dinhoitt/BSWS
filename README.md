# BSWS

**Better Scores, Worse Speech: An Objective-Subjective Ranking Reversal in BigVGAN-Style Discriminator Ablations**

Research project page, fixed listening audio, aggregate listening-test results, and objective evaluation resources.

## Website

The static website is in `docs/`. Enable GitHub Pages for `main` → `/docs`.
The intended URL is `https://dinhoitt.github.io/BSWS/`.
There is no build step, database, response collection, analytics script, or external font dependency.

For a local preview, serve `docs/` using any static HTTP server.

## Contents

- `docs/audio/`: the 120 WAVs actually used in listening evaluation (20 utterances × 6 conditions).
- `docs/data/audio_manifest.csv`: portable paths, corpus IDs, audio properties, reported checkpoint steps and SHA-256 hashes.
- `docs/data/listening_summary.csv`: MOS, paired MOS differences and normalized CMOS with listener-level 95% intervals.
- `docs/data/objective_paired.csv`: objective comparisons on the same 20 utterances, including UTMOS and paired-utterance bootstrap intervals.
- `docs/data/objective_per_utterance.csv`: individual automatic measurements.
- `evaluation/`: portable objective-evaluation code. See its README for dependencies and execution.
- `docs/downloads/`: downloadable audio and evaluation-code bundles.

## Data availability and privacy

This release publishes aggregate listening-test results, **not participant-level responses**. Prolific identifiers, session identifiers, correspondence, private access links and local machine paths must not be added to this repository or its history. Participant-level release requires a separate review of consent and de-identification.

The listening results are a fixed snapshot verified on 2026-09-22. Objective matched-stimulus results are the archived 2026-09-23 reanalysis. Neither is a live database query.

## Audio attribution

Reference speech comes from LibriTTS (Zen et al., 2019), [OpenSLR 60](https://www.openslr.org/60/), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Reference clips are selected from `test-clean`; the synthesized conditions are model reconstructions, not original corpus recordings. Public WAVs are copied without further normalization or cropping from the fixed listening assets. See `docs/data/README.md` and the manifest for details.

Code dependencies retain their upstream licenses. No blanket license for third-party code, learned weights, or all repository contents is implied by this README.

## Paper availability sentence

After the page and downloads are publicly accessible:

```latex
Audio samples, listening-test summary results, and evaluation code
are available at \url{https://dinhoitt.github.io/BSWS/}.
```

Do not replace “summary results” with “responses” unless participant-level ratings have actually been released.
