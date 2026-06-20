# Notebooks

These notebooks demonstrate the Python extension API with matplotlib plots.

Install the local package and notebook dependencies from the repository root:

```bash
python -m pip install maturin notebook matplotlib
maturin develop
jupyter notebook notebooks
```

Available notebooks:

- `01_getting_started_features.ipynb`: synthetic signal, MFCC/GFCC, Mel spectrogram, and spectral descriptors.
- `02_filter_banks.ipynb`: Mel, linear, Bark, and gammatone filter-bank comparison.
- `03_pitch_tracking.ipynb`: YIN pitch tracking and dominant-frequency comparison.
- `04_cochleagram.ipynb`: cochleagram generation and visualization.
