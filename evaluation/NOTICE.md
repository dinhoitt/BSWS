# Attribution and release scope

The evaluation modules accompany the BSWS research project. No blanket license
is inferred for the author's code or study data; the repository owner should
choose an explicit code/data license before describing this as an open-source
or open-data release. Public availability alone is not a license grant.

Third-party dependencies are installed separately and retain their respective
licenses. No third-party model weights or vendored dependency source files are
included. Key implementations are:

- [librosa](https://github.com/librosa/librosa), audio preprocessing.
- [pysptk](https://github.com/r9y9/pysptk) and [SPTK](https://github.com/sp-nitech/SPTK), mel-cepstral analysis.
- [fastdtw](https://github.com/slaypni/fastdtw), approximate dynamic time warping.
- [scikit-image](https://github.com/scikit-image/scikit-image), SSIM.
- [auraloss](https://github.com/csteinmetz1/auraloss), multi-resolution STFT loss.
- [python-pesq](https://github.com/ludlows/PESQ), PESQ wrapper; review its licensing and underlying PESQ terms.
- [torchcrepe](https://github.com/maxrmorrison/torchcrepe), pitch and periodicity.
- [scikit-learn](https://github.com/scikit-learn/scikit-learn), voicing F1.
- [SpeechMOS](https://github.com/tarepan/SpeechMOS/tree/v1.2.0) and
  [UTMOS22](https://github.com/sarulab-speech/UTMOS22), optional learned quality prediction.
- [NumPy](https://github.com/numpy/numpy), [SciPy](https://github.com/scipy/scipy),
  [PyTorch](https://github.com/pytorch/pytorch), [torchaudio](https://github.com/pytorch/audio),
  and [SoundFile](https://github.com/bastibe/python-soundfile).

The SpeechMOS code at the pinned revision reports an MIT license. Its
repository, downloaded assets, and underlying models may have additional
notices; consult the upstream sources. This file is attribution, not a
replacement for upstream license terms.
