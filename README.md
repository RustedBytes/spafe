# Spafe

Native Rust implementation of the core `spafe` audio feature extraction APIs.

The crate provides filter banks, spectrograms, cepstral features, spectral
descriptors, frequency estimators, cochleagram generation, preprocessing helpers,
and deterministic SVG visualization helpers.

## Build

```bash
cargo build
```

The crate also has an opt-in portable SIMD path for hot numeric kernels. Because
`std::simd` is still unstable, this feature currently requires nightly Rust:

```bash
cargo +nightly build --features portable-simd
```

## Python Bindings

The crate exposes a PyO3 extension module through maturin. Python functions use
plain Python lists for one-dimensional signals and nested lists for matrices.

Install locally in a virtual environment:

```bash
python -m pip install maturin
maturin develop
```

Build a wheel:

```bash
python -m pip install build
python -m build --wheel
```

Basic Python usage:

```python
import math
import spafe

fs = 16_000
sig = [math.sin(2.0 * math.pi * 440.0 * n / fs) for n in range(fs)]
opts = spafe.FeatureOptions(fs=fs, nfft=256, nfilts=24)

ceps = spafe.mfcc(sig, opts)
fbanks, centers = spafe.mel_filter_banks(spafe.FilterBankOptions(nfft=256))

print(len(ceps), len(ceps[0]))
print(len(fbanks), len(centers))
```

## Test

```bash
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings -D missing_docs" cargo doc --no-deps
```

## Examples

Generate a filter-bank SVG:

```bash
cargo run --example fbanks
open target/fbanks.svg
```

Generate a cochleagram SVG:

```bash
cargo run --example cochleagram
open target/cochleagram.svg
```

Python examples live in `examples/python`, use matplotlib, and write outputs to
`target/python-examples`:

```bash
python -m pip install matplotlib
python examples/python/features.py
python examples/python/fbanks.py
python examples/python/pitch.py
python examples/python/cochleagram.py
```

Interactive notebooks live in `notebooks`:

```bash
python -m pip install notebook matplotlib
jupyter notebook notebooks
```

They cover feature extraction, filter banks, pitch tracking, cochleagrams,
spectrogram variants, conversion helpers, and option tuning.
The real-audio notebook uses `notebooks/sample_uk.opus` and requires `ffmpeg`.

## Basic Usage

```rust
use spafe::prelude::*;

fn main() -> spafe::Result<()> {
    let fs = 16_000;
    let signal = vec![0.0; fs];
    let opts = FeatureOptions {
        fs,
        ..Default::default()
    };

    let coeffs = mfcc(&signal, &opts)?;
    println!("frames={}, coeffs={}", coeffs.nrows(), coeffs.ncols());

    let coch = cochleagram(&signal, &CochleagramOptions {
        signal_size: signal.len(),
        sr: fs,
        env_sr: 400,
        ..Default::default()
    })?;
    println!("filters={}, samples={}", coch.cochleagram.nrows(), coch.cochleagram.ncols());
    Ok(())
}
```
