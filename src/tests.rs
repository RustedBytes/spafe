use super::*;

fn sine() -> Vec<f64> {
    let fs = 16_000.0;
    (0..16_000)
        .map(|n| (2.0 * std::f64::consts::PI * 440.0 * n as f64 / fs).sin())
        .collect()
}

#[test]
fn converters_roundtrip() {
    let mel = utils::converters::hz2mel(1000.0, MelConversionApproach::Oshaghnessy);
    let hz = utils::converters::mel2hz(mel, MelConversionApproach::Oshaghnessy);
    assert!((hz - 1000.0).abs() < 1e-9);

    let mel = utils::converters::hz2mel(1000.0, MelConversionApproach::Lindsay);
    let hz = utils::converters::mel2hz(mel, MelConversionApproach::Lindsay);
    assert!((hz - 1000.0).abs() < 1e-9);

    let erb = utils::converters::hz2erb(1000.0, ErbConversionApproach::Glasberg);
    let hz = utils::converters::erb2hz(erb, ErbConversionApproach::Glasberg);
    assert!((hz - 1000.0).abs() < 1e-9);

    for approach in [
        BarkConversionApproach::Wang,
        BarkConversionApproach::Tjomov,
        BarkConversionApproach::Schroeder,
        BarkConversionApproach::Terhardt,
        BarkConversionApproach::Zwicker,
        BarkConversionApproach::Traunmueller,
    ] {
        let bark = utils::converters::hz2bark(1000.0, approach);
        let hz = utils::converters::bark2hz(bark, approach);
        assert!(
            (hz - 1000.0).abs() < 1e-9,
            "{approach:?} did not roundtrip: {hz}"
        );
    }
}

#[test]
fn filter_bank_shapes() {
    let opts = FilterBankOptions::default();
    let (mel, centers) =
        fbanks::mel_filter_banks(&opts, MelConversionApproach::Oshaghnessy).unwrap();
    assert_eq!(mel.dim(), (24, 257));
    assert_eq!(centers.len(), 24);

    let (bark, centers) = fbanks::bark_filter_banks(&opts, BarkConversionApproach::Wang).unwrap();
    assert_eq!(bark.dim(), (24, 257));
    assert_eq!(centers.len(), 24);

    let (linear, centers) = fbanks::linear_filter_banks(&opts).unwrap();
    assert_eq!(linear.dim(), (24, 257));
    assert_eq!(centers.len(), 24);

    let (inverse, centers) =
        fbanks::inverse_mel_filter_banks(&opts, MelConversionApproach::Oshaghnessy).unwrap();
    assert_eq!(inverse.dim(), (24, 257));
    assert_eq!(centers.len(), 24);

    let (gammatone, centers) =
        fbanks::gammatone_filter_banks(&opts, 4, ErbConversionApproach::Glasberg).unwrap();
    assert_eq!(gammatone.dim(), (24, 257));
    assert_eq!(centers.len(), 24);
    assert!(gammatone.iter().all(|value| value.is_finite()));
}

#[test]
fn invalid_filter_bank_options_return_errors() {
    let low_freq = FilterBankOptions {
        low_freq: -1.0,
        ..FilterBankOptions::default()
    };
    assert_eq!(
        fbanks::mel_filter_banks(&low_freq, MelConversionApproach::Oshaghnessy).unwrap_err(),
        SpafeError::LowFrequency
    );

    let high_freq = FilterBankOptions {
        high_freq: Some(9_000.0),
        ..FilterBankOptions::default()
    };
    assert_eq!(
        fbanks::linear_filter_banks(&high_freq).unwrap_err(),
        SpafeError::HighFrequency
    );
}

#[test]
fn preprocessing_and_cepstral_helpers_are_deterministic() {
    let emphasized = utils::preprocessing::pre_emphasis(&[1.0, 2.0, 4.0], 0.5);
    assert_eq!(emphasized, vec![1.0, 1.5, 3.0]);

    let frames = utils::preprocessing::stride_trick(&[1.0, 2.0, 3.0, 4.0], 2, 1).unwrap();
    assert_eq!(frames, ndarray::arr2(&[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]));

    assert_eq!(
        utils::preprocessing::framing(&[1.0, 2.0], 10, 0.3, 0.1).unwrap_err(),
        SpafeError::SignalTooShort
    );
    assert_eq!(
        utils::preprocessing::framing(&[1.0, 2.0, 3.0], 10, 0.1, 0.2).unwrap_err(),
        SpafeError::WindowLength
    );

    let hanning = utils::preprocessing::window_values(5, WindowType::Hanning);
    assert!(hanning[0].abs() < 1e-12);
    assert!((hanning[2] - 1.0).abs() < 1e-12);
    assert!(hanning[4].abs() < 1e-12);

    let scaled = utils::filters::scale_fbank(Scale::Ascendant, 4);
    assert_eq!(scaled.column(0).to_vec(), vec![0.25, 0.5, 0.75, 1.0]);

    let ceps = ndarray::arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let centered = utils::cepstral::normalize_ceps(&ceps, Normalization::MeanSubtraction);
    assert!((centered.column(0).sum()).abs() < 1e-12);
    assert!((centered.column(1).sum()).abs() < 1e-12);
    assert!((centered.column(2).sum()).abs() < 1e-12);

    let lifted = utils::cepstral::lifter_ceps(&ceps, 2);
    assert_eq!(lifted[(0, 0)], 1.0);
    assert_eq!(lifted[(0, 1)], 2.0);
    assert_eq!(lifted[(0, 2)], 12.0);

    let deltas = utils::cepstral::deltas(&ceps, 3);
    assert_eq!(deltas.dim(), ceps.dim());
    assert!(deltas.iter().all(|value| value.is_finite()));
}

#[test]
fn mfcc_shape() {
    let sig = sine();
    let opts = FeatureOptions::default();
    let ceps = features::mfcc(&sig, &opts).unwrap();
    assert_eq!(ceps.ncols(), 13);
    assert!(ceps.nrows() > 0);
}

#[test]
fn spectrogram_variants_have_expected_shapes() {
    let sig = sine();
    let opts = FeatureOptions {
        nfft: 256,
        nfilts: 16,
        window: SlidingWindow {
            win_len: 0.025,
            win_hop: 0.010,
            win_type: WindowType::Hamming,
        },
        ..FeatureOptions::default()
    };

    let outputs = [
        features::mel_spectrogram(&sig, &opts).unwrap(),
        features::linear_spectrogram(&sig, &opts).unwrap(),
        features::bark_spectrogram(&sig, &opts).unwrap(),
        features::erb_spectrogram(&sig, &opts).unwrap(),
    ];

    for output in outputs {
        assert_eq!(output.features.ncols(), opts.nfilts);
        assert_eq!(output.fft_magnitude.ncols(), opts.nfft / 2 + 1);
        assert_eq!(output.features.nrows(), output.fft_magnitude.nrows());
        assert!(output.features.iter().all(|value| value.is_finite()));
    }
}

#[test]
fn yin_runs() {
    let sig = sine();
    let (pitches, _, _, _) =
        frequencies::compute_yin(&sig, 16_000, 0.03, 0.015, 50.0, 1000.0, 0.1).unwrap();
    assert!(!pitches.is_empty());
}

#[test]
fn frequency_helpers_find_expected_periods_and_bins() {
    let diff = frequencies::compute_difference(&[1.0, 2.0, 4.0], 4);
    assert_eq!(diff.to_vec(), vec![0.0, 5.0, 9.0]);

    let cmnd = frequencies::compute_cmnd(diff.as_slice().unwrap(), 3);
    assert_eq!(cmnd[0], 1.0);
    assert!((cmnd[1] - 1.0).abs() < 1e-12);
    assert!((cmnd[2] - (18.0 / 14.0)).abs() < 1e-12);

    let pitch = frequencies::get_pitch(&[1.0, 0.4, 0.08, 0.04, 0.2], 1, 5, 0.1);
    assert_eq!(pitch, 3.0);

    let sig = sine();
    let dominant = frequencies::get_dominant_frequencies(
        &sig,
        16_000,
        512,
        0.025,
        0.010,
        WindowType::Hamming,
        true,
    )
    .unwrap();
    assert!(!dominant.is_empty());
    assert!((dominant[0] - 437.5).abs() < 1.0);
}

#[test]
fn feature_families_run() {
    let sig = sine();
    let opts = FeatureOptions {
        nfft: 256,
        nfilts: 24,
        window: SlidingWindow {
            win_len: 0.025,
            win_hop: 0.02,
            win_type: WindowType::Hamming,
        },
        ..FeatureOptions::default()
    };

    let funcs: Vec<Matrix> = vec![
        features::imfcc(&sig, &opts).unwrap(),
        features::lfcc(&sig, &opts).unwrap(),
        features::bfcc(&sig, &opts).unwrap(),
        features::gfcc(&sig, &opts).unwrap(),
        features::msrcc(&sig, &opts, -1.0 / 7.0).unwrap(),
        features::ngcc(&sig, &opts).unwrap(),
        features::psrcc(&sig, &opts, -1.0 / 7.0).unwrap(),
        features::pncc(&sig, &opts, 2.0).unwrap(),
        features::lpcc(&sig, &opts).unwrap(),
        features::plp(&sig, &opts, false).unwrap(),
        features::rplp(&sig, &opts).unwrap(),
        features::cqcc(
            &sig,
            &opts,
            &CqccOptions {
                number_of_octaves: 5,
                number_of_bins_per_octave: 12,
                ..CqccOptions::default()
            },
        )
        .unwrap(),
    ];

    for mat in funcs {
        assert_eq!(mat.ncols(), 13);
        assert!(mat.nrows() > 0);
    }
}

#[test]
fn spectral_descriptors_are_finite_and_shaped() {
    let sig = sine();
    let feats = spfeats::extract_feats_with_nfft(&sig, 16_000, 256);
    assert!(feats.spectral_centroid.is_finite());
    assert!(feats.spectral_skewness.is_finite());
    assert!(feats.spectral_kurtosis.is_finite());
    assert!(feats.spectral_entropy.is_finite());
    assert_eq!(feats.spectral_spread.len(), 129);
    assert!((0.0..=1.0).contains(&feats.spectral_flatness));
    assert_eq!(feats.spectral_rolloff.len(), 129);
    assert!(feats.spectral_flux.is_finite());
    assert!(feats.spectral_mean.re.is_finite());
    assert!(feats.spectral_rms.norm().is_finite());
    assert!(feats.spectral_std.is_finite());
    assert!(feats.spectral_variance.is_finite());
}

#[test]
fn visualization_renders_svg() {
    let opts = FilterBankOptions {
        nfilts: 4,
        nfft: 64,
        fs: 8_000,
        high_freq: Some(4_000.0),
        ..FilterBankOptions::default()
    };
    let (fbanks, centers) =
        fbanks::linear_filter_banks(&opts).expect("linear filter bank should build");
    let freqs = (0..(opts.nfft / 2 + 1))
        .map(|idx| idx as f64 * opts.fs as f64 / opts.nfft as f64)
        .collect::<Vec<_>>();

    let ticks = utils::vis::tick_function(&[0.0, 2000.0], "mel");
    assert_eq!(ticks[0], "0.0");
    assert_ne!(ticks[1], "2000.0");

    let fbank_svg = utils::vis::show_fbanks(
        &fbanks,
        centers.as_slice().unwrap(),
        &freqs,
        "Linear Filter Bank",
        "Weight",
        "Frequency / Hz",
        "Frequency / mel",
        "lin",
        true,
    )
    .unwrap();
    assert!(fbank_svg.starts_with("<svg"));
    assert!(fbank_svg.contains("Linear Filter Bank"));

    let spectrogram_svg = utils::vis::show_spectrogram(
        &fbanks,
        8_000,
        0.0,
        1.0,
        0.0,
        4.0,
        80.0,
        "Time (s)",
        "Frequency (Hz)",
        "Spectrogram",
        true,
    )
    .unwrap();
    assert!(spectrogram_svg.contains("<rect"));

    let features_svg =
        utils::vis::show_features(&fbanks, "Features", "Feature Index", "Frame Index").unwrap();
    assert!(features_svg.contains("Features"));
}

#[test]
fn erb_cos_filters_match_chcochleagram_shape() {
    let opts = cochleagram::ErbCosFilterOptions {
        n: 40,
        low_lim: 50.0,
        high_lim: 10_000.0,
        sample_factor: 4,
        full_filter: false,
        ..Default::default()
    };
    let filters = cochleagram::erb_cos_filter_bank(40_000, 20_000, None, &opts).unwrap();
    assert_eq!(filters.filters.dim(), (171, 20_001));
    assert_eq!(filters.center_freqs.len(), 171);

    let padded = cochleagram::erb_cos_filter_bank(40_000, 20_000, Some(2), &opts).unwrap();
    assert_eq!(padded.filters.dim(), (171, 40_001));
}

#[test]
fn cochleagram_pipeline_runs() {
    let signal_size = 4096;
    let sr = 16_000;
    let sig = (0..signal_size)
        .map(|idx| (2.0 * std::f64::consts::PI * 440.0 * idx as f64 / sr as f64).sin())
        .collect::<Vec<_>>();
    let opts = cochleagram::CochleagramOptions {
        signal_size,
        sr,
        env_sr: 400,
        filter: cochleagram::ErbCosFilterOptions {
            n: 8,
            low_lim: 50.0,
            high_lim: 6_000.0,
            sample_factor: 2,
            full_filter: false,
            ..Default::default()
        },
        downsampling: cochleagram::DownsamplingMode::SincWithKaiserWindow {
            window_size: 129,
            padding: None,
        },
        compression: Some(cochleagram::CompressionMode::Power {
            scale: 1.0,
            offset: 1e-8,
            power: 0.3,
        }),
        ..Default::default()
    };

    let out = cochleagram::cochleagram(&sig, &opts).unwrap();
    assert_eq!(out.filter_bank.filters.nrows(), 21);
    assert_eq!(out.latents.envelopes.nrows(), 21);
    assert_eq!(out.cochleagram.nrows(), 21);
    assert!(out.cochleagram.ncols() > 0);
    assert!(out.cochleagram.iter().all(|value| value.is_finite()));
    assert!(out.cochleagram.iter().all(|value| *value >= 0.0));
}

#[test]
fn cochleagram_modes_and_errors_are_covered() {
    let signal_size = 1024;
    let sr = 8_000;
    let sig = (0..signal_size)
        .map(|idx| (2.0 * std::f64::consts::PI * 330.0 * idx as f64 / sr as f64).sin())
        .collect::<Vec<_>>();
    let base = cochleagram::CochleagramOptions {
        signal_size,
        sr,
        env_sr: 1_000,
        filter: cochleagram::ErbCosFilterOptions {
            n: 4,
            low_lim: 80.0,
            high_lim: 3_000.0,
            sample_factor: 2,
            full_filter: false,
            ..Default::default()
        },
        downsampling: cochleagram::DownsamplingMode::HannPooling1d {
            window_size: 8,
            padding: 0,
            normalize: true,
        },
        compression: Some(cochleagram::CompressionMode::Linear {
            scale: 2.0,
            offset: 0.1,
        }),
        ..Default::default()
    };

    for envelope in [
        cochleagram::EnvelopeMode::AbsSubbands,
        cochleagram::EnvelopeMode::RectifySubbands,
    ] {
        let opts = cochleagram::CochleagramOptions {
            envelope,
            ..base.clone()
        };
        let out = cochleagram::cochleagram(&sig, &opts).unwrap();
        assert_eq!(out.cochleagram.nrows(), out.filter_bank.filters.nrows());
        assert!(out.cochleagram.iter().all(|value| value.is_finite()));
        assert!(out.cochleagram.iter().all(|value| *value >= 0.0));
    }

    let invalid_len = cochleagram::cochleagram(&sig[..512], &base).unwrap_err();
    assert!(matches!(invalid_len, SpafeError::InvalidParameter(_)));

    let invalid_downsample = cochleagram::CochleagramOptions {
        env_sr: 3_000,
        ..base.clone()
    };
    let err = cochleagram::cochleagram(&sig, &invalid_downsample).unwrap_err();
    assert!(matches!(err, SpafeError::InvalidParameter(_)));
}

#[test]
fn cochleagram_helpers_run() {
    let erb = cochleagram::freq2erb(1000.0);
    let hz = cochleagram::erb2freq(erb);
    assert!((hz - 1000.0).abs() < 1e-9);

    let values = cochleagram::make_cosine_filter(&[10.0, 20.0, 30.0, 40.0], 15.0, 35.0, false);
    assert_eq!(values.len(), 2);

    let input = ndarray::arr2(&[[1e-2, 1e-4, 0.0, -1.0]]);
    let compressed = cochleagram::apply_compression(
        &input,
        cochleagram::CompressionMode::ClippedGradPower {
            scale: 1.0,
            offset: 1e-8,
            power: 0.3,
            clip_value: 100.0,
        },
    );
    assert!(compressed[(0, 0)] > compressed[(0, 1)]);
    assert_eq!(compressed[(0, 2)], compressed[(0, 3)]);

    let padding = cochleagram::calculate_same_padding(40_000, 1001, 100, 1);
    assert_eq!(padding, (450, 451));
}
