# Notebooks

These notebooks demonstrate the Python extension API with matplotlib plots.

Install the local package and notebook dependencies from the repository root:

```bash
python -m pip install maturin notebook matplotlib numpy
maturin develop
jupyter notebook notebooks
```

The real-audio notebook also requires the `ffmpeg` command-line tool to decode
the Opus sample.

Available notebooks:

- `01_getting_started_features.ipynb`: synthetic signal, MFCC/GFCC, Mel spectrogram, and spectral descriptors.
- `02_filter_banks.ipynb`: Mel, linear, Bark, and gammatone filter-bank comparison.
- `03_pitch_tracking.ipynb`: YIN pitch tracking and dominant-frequency comparison.
- `04_cochleagram.ipynb`: cochleagram generation and visualization.
- `05_advanced_cepstral_features.ipynb`: IMFCC, LFCC, BFCC, MSRCC, NGCC, PSRCC, PNCC, LPCC, PLP, RPLP, and CQCC.
- `06_spectrogram_variants.ipynb`: Mel, linear, Bark, and ERB spectrograms plus inverse Mel filter banks.
- `07_converters_and_options.ipynb`: Hertz/Mel/Bark/ERB converters and option tuning examples.
- `08_real_audio_sample_uk.ipynb`: load `sample_uk.opus`, then compute features, pitch, cochleagram, and descriptors on real audio.
- `09_performance.ipynb`: benchmark representative feature extractors, signal-duration scaling, and core operations.
- `10_visualization_cookbook.ipynb`: reusable matplotlib recipes for waveforms, filter banks, feature heatmaps, pitch tracks, cochleagrams, and dashboards.
- `11_export_workflow.ipynb`: export features, pitch tracks, metadata, CSV files, and NumPy archives for downstream workflows.
- `12_feature_comparison_classification.ipynb`: compare feature families on a small classification workflow with PCA plots and a nearest-centroid classifier.
- `13_parameter_sensitivity.ipynb`: inspect how `nfft`, `nfilts`, window settings, normalization, liftering, scaling, and frequency bounds affect outputs.
