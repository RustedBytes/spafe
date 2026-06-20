use spafe::prelude::*;
use std::fs;

fn main() -> spafe::Result<()> {
    let opts = FilterBankOptions {
        nfilts: 24,
        nfft: 512,
        fs: 16_000,
        high_freq: Some(8_000.0),
        ..Default::default()
    };

    let (fbanks, centers) = linear_filter_banks(&opts)?;

    let freqs: Vec<f64> = (0..opts.nfft / 2 + 1)
        .map(|i| i as f64 * opts.fs as f64 / opts.nfft as f64)
        .collect();

    let svg = show_fbanks(
        &fbanks,
        centers.as_slice().unwrap(),
        &freqs,
        "Linear Filter Bank",
        "Weight",
        "Frequency / Hz",
        "Frequency / Hz",
        "lin",
        true,
    )?;

    fs::write("target/fbanks.svg", svg).unwrap();
    Ok(())
}
