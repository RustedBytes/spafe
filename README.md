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
