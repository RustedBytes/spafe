use spafe::prelude::*;
use std::fs;

fn main() -> spafe::Result<()> {
    let signal_size = 4096;
    let sr = 16_000;
    let signal = (0..signal_size)
        .map(|idx| (2.0 * std::f64::consts::PI * 440.0 * idx as f64 / sr as f64).sin())
        .collect::<Vec<_>>();

    let opts = CochleagramOptions {
        signal_size,
        sr,
        env_sr: 400,
        filter: ErbCosFilterOptions {
            n: 16,
            low_lim: 50.0,
            high_lim: 6_000.0,
            sample_factor: 2,
            ..Default::default()
        },
        downsampling: DownsamplingMode::SincWithKaiserWindow {
            window_size: 129,
            padding: None,
        },
        ..Default::default()
    };

    let output = cochleagram(&signal, &opts)?;
    let svg = show_features(
        &output.cochleagram.t().to_owned(),
        "Cochleagram",
        "Filter Index",
        "Frame Index",
    )?;
    fs::write("target/cochleagram.svg", svg).unwrap();
    Ok(())
}
