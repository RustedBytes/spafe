use ndarray::{Array1, Array2, Axis};
use thiserror::Error;

/// Two-dimensional `f64` matrix used for frames, filter banks, spectrograms, and features.
pub type Matrix = Array2<f64>;
/// One-dimensional `f64` vector used for frequency lists, pitch tracks, and energies.
pub type Vector = Array1<f64>;
/// Crate-local result type returned by fallible `spafe` operations.
pub type Result<T> = std::result::Result<T, SpafeError>;

/// Errors returned by invalid feature, filter-bank, and preprocessing parameters.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum SpafeError {
    /// `low_freq` must not be negative.
    #[error("low_freq must be non-negative")]
    LowFrequency,
    /// `high_freq` must not exceed the Nyquist frequency.
    #[error("high_freq must be less than or equal to fs / 2")]
    HighFrequency,
    /// Window length must be greater than or equal to the hop length.
    #[error("win_len must be greater than or equal to win_hop")]
    WindowLength,
    /// The number of filters must be greater than or equal to the requested cepstra.
    #[error("nfilts must be greater than or equal to num_ceps")]
    FilterCount,
    /// The signal is too short for the requested frame configuration.
    #[error("signal is too short for the requested window")]
    SignalTooShort,
    /// Generic parameter error for checks without a dedicated variant.
    #[error("invalid parameter: {0}")]
    InvalidParameter(&'static str),
}

/// Filter-bank scaling strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Scale {
    /// Increase filter weights from low to high filter index.
    Ascendant,
    /// Decrease filter weights from low to high filter index.
    Descendant,
    /// Keep all filter weights equal.
    #[default]
    Constant,
}

/// Window function applied during frame blocking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WindowType {
    /// Hann window.
    Hanning,
    /// Bartlett triangular window.
    Bartlet,
    /// Kaiser window with the same beta used by the Python implementation.
    Kaiser,
    /// Blackman window.
    Blackman,
    /// Hamming window.
    #[default]
    Hamming,
}

/// Frame-blocking configuration used before spectral feature extraction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SlidingWindow {
    /// Window length in seconds.
    pub win_len: f64,
    /// Hop length in seconds.
    pub win_hop: f64,
    /// Window function applied to every frame.
    pub win_type: WindowType,
}

impl Default for SlidingWindow {
    fn default() -> Self {
        Self {
            win_len: 0.025,
            win_hop: 0.010,
            win_type: WindowType::Hamming,
        }
    }
}

/// Cepstral normalization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Normalization {
    /// Column-wise mean subtraction followed by global variance scaling.
    MeanVariance,
    /// Column-wise mean subtraction.
    MeanSubtraction,
    /// Global variance scaling.
    Variance,
    /// Global mean normalization using the matrix value range.
    Mean,
}

/// Formula used to convert between Hertz and Mel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MelConversionApproach {
    /// O'Shaughnessy conversion formula.
    #[default]
    Oshaghnessy,
    /// Lindsay conversion formula.
    Lindsay,
}

/// Formula used to convert between Hertz and Bark.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BarkConversionApproach {
    /// Wang conversion formula.
    #[default]
    Wang,
    /// Tjomov conversion formula.
    Tjomov,
    /// Schroeder conversion formula.
    Schroeder,
    /// Terhardt conversion formula.
    Terhardt,
    /// Zwicker conversion formula.
    Zwicker,
    /// Traunmueller conversion formula.
    Traunmueller,
}

/// Formula used to convert between Hertz and ERB.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ErbConversionApproach {
    /// Glasberg and Moore conversion formula.
    #[default]
    Glasberg,
}

/// Options shared by Mel, linear, Bark, and gammatone filter-bank builders.
#[derive(Debug, Clone)]
pub struct FilterBankOptions {
    /// Number of filters to generate.
    pub nfilts: usize,
    /// FFT size; filter-bank columns correspond to `nfft / 2 + 1` bins.
    pub nfft: usize,
    /// Sampling rate in Hertz.
    pub fs: usize,
    /// Lowest frequency covered by the filters.
    pub low_freq: f64,
    /// Highest frequency covered by the filters; defaults to `fs / 2`.
    pub high_freq: Option<f64>,
    /// Optional per-filter scaling.
    pub scale: Scale,
}

impl Default for FilterBankOptions {
    fn default() -> Self {
        Self {
            nfilts: 24,
            nfft: 512,
            fs: 16_000,
            low_freq: 0.0,
            high_freq: None,
            scale: Scale::Constant,
        }
    }
}

/// Options shared by cepstral feature extractors and spectrogram builders.
#[derive(Debug, Clone)]
pub struct FeatureOptions {
    /// Sampling rate in Hertz.
    pub fs: usize,
    /// Number of cepstral coefficients to return.
    pub num_ceps: usize,
    /// Whether to pre-emphasize the signal before framing.
    pub pre_emph: bool,
    /// Pre-emphasis coefficient.
    pub pre_emph_coeff: f64,
    /// Frame-blocking and windowing configuration.
    pub window: SlidingWindow,
    /// Number of filters to use in the perceptual filter bank.
    pub nfilts: usize,
    /// FFT size.
    pub nfft: usize,
    /// Lowest analysis frequency.
    pub low_freq: f64,
    /// Highest analysis frequency; defaults to `fs / 2`.
    pub high_freq: Option<f64>,
    /// Optional per-filter scaling.
    pub scale: Scale,
    /// DCT type used to convert log/power spectra to cepstra.
    pub dct_type: usize,
    /// Whether to replace the first cepstral coefficient with frame energy.
    pub use_energy: bool,
    /// Optional cepstral liftering value.
    pub lifter: Option<i32>,
    /// Optional cepstral normalization mode.
    pub normalize: Option<Normalization>,
}

impl Default for FeatureOptions {
    fn default() -> Self {
        Self {
            fs: 16_000,
            num_ceps: 13,
            pre_emph: true,
            pre_emph_coeff: 0.97,
            window: SlidingWindow::default(),
            nfilts: 24,
            nfft: 512,
            low_freq: 0.0,
            high_freq: None,
            scale: Scale::Constant,
            dct_type: 2,
            use_energy: false,
            lifter: None,
            normalize: None,
        }
    }
}

/// Additional options for constant-Q cepstral coefficient extraction.
#[derive(Debug, Clone)]
pub struct CqccOptions {
    /// Number of octaves in the constant-Q frequency grid.
    pub number_of_octaves: usize,
    /// Number of constant-Q bins per octave.
    pub number_of_bins_per_octave: usize,
    /// Ratio used when resampling the log constant-Q spectrum before the DCT.
    pub resampling_ratio: f64,
    /// Threshold below which constant-Q kernel coefficients are zeroed.
    pub spectral_threshold: f64,
    /// Base frequency of the constant-Q transform.
    pub f0: f64,
    /// Constant-Q rate multiplier.
    pub q_rate: f64,
}

impl Default for CqccOptions {
    fn default() -> Self {
        Self {
            number_of_octaves: 7,
            number_of_bins_per_octave: 24,
            resampling_ratio: 1.0,
            spectral_threshold: 0.005,
            f0: 120.0,
            q_rate: 1.0,
        }
    }
}

/// Spectrogram output containing filtered energies and the underlying FFT magnitudes.
#[derive(Debug, Clone)]
pub struct SpectrogramOutput {
    /// Filter-bank energies, one row per frame.
    pub features: Matrix,
    /// Magnitude FFT, one row per frame and one column per FFT bin.
    pub fft_magnitude: Matrix,
}

pub(crate) fn checked_high_freq(low_freq: f64, high_freq: Option<f64>, fs: usize) -> Result<f64> {
    if low_freq < 0.0 {
        return Err(SpafeError::LowFrequency);
    }
    let high = high_freq.unwrap_or(fs as f64 / 2.0);
    if high > fs as f64 / 2.0 {
        return Err(SpafeError::HighFrequency);
    }
    Ok(high)
}

pub(crate) fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    match n {
        0 => Vec::new(),
        1 => vec![start],
        _ => {
            let step = (end - start) / (n - 1) as f64;
            (0..n).map(|i| start + i as f64 * step).collect()
        }
    }
}

pub(crate) fn dct_rows(x: &Matrix, dct_type: usize, out_cols: usize) -> Matrix {
    let cols = x.ncols();
    let keep = out_cols.min(cols);
    let mut out = Array2::<f64>::zeros((x.nrows(), keep));

    for (r, row) in x.axis_iter(Axis(0)).enumerate() {
        for k in 0..keep {
            let value = match dct_type {
                1 => {
                    if cols == 1 {
                        row[0]
                    } else {
                        let mut sum = row[0]
                            + if k % 2 == 0 {
                                row[cols - 1]
                            } else {
                                -row[cols - 1]
                            };
                        for n in 1..cols - 1 {
                            sum += 2.0
                                * row[n]
                                * (std::f64::consts::PI * k as f64 * n as f64 / (cols - 1) as f64)
                                    .cos();
                        }
                        let scale = if k == 0 || k == cols - 1 {
                            (1.0 / (4.0 * (cols - 1) as f64)).sqrt()
                        } else {
                            (1.0 / (2.0 * (cols - 1) as f64)).sqrt()
                        };
                        sum * scale
                    }
                }
                3 => {
                    let mut sum = row[0] / 2.0_f64.sqrt();
                    for n in 1..cols {
                        sum += row[n]
                            * (std::f64::consts::PI * n as f64 * (k as f64 + 0.5) / cols as f64)
                                .cos();
                    }
                    sum * (2.0 / cols as f64).sqrt()
                }
                4 => {
                    let mut sum = 0.0;
                    for n in 0..cols {
                        sum += row[n]
                            * (std::f64::consts::PI * (n as f64 + 0.5) * (k as f64 + 0.5)
                                / cols as f64)
                                .cos();
                    }
                    sum * (2.0 / cols as f64).sqrt()
                }
                _ => {
                    let mut sum = 0.0;
                    for n in 0..cols {
                        sum += row[n]
                            * (std::f64::consts::PI * (n as f64 + 0.5) * k as f64 / cols as f64)
                                .cos();
                    }
                    let ck = if k == 0 { 1.0 / 2.0_f64.sqrt() } else { 1.0 };
                    sum * (2.0 / cols as f64).sqrt() * ck
                }
            };
            out[(r, k)] = value;
        }
    }
    out
}

pub(crate) fn apply_post_processing(mut ceps: Matrix, opts: &FeatureOptions) -> Matrix {
    if let Some(lift) = opts.lifter {
        ceps = crate::utils::cepstral::lifter_ceps(&ceps, lift);
    }
    if let Some(normalize) = opts.normalize {
        ceps = crate::utils::cepstral::normalize_ceps(&ceps, normalize);
    }
    ceps
}

pub(crate) fn replace_zero(x: f64) -> f64 {
    if x == 0.0 { f64::EPSILON } else { x }
}

pub(crate) fn cepstral_from_spectrogram(
    spectrogram: SpectrogramOutput,
    opts: &FeatureOptions,
    transform: impl Fn(f64) -> f64,
) -> Matrix {
    let mapped = spectrogram.features.mapv(|v| transform(replace_zero(v)));
    let mut ceps = dct_rows(&mapped, opts.dct_type, opts.num_ceps);

    if opts.use_energy && ceps.ncols() > 0 {
        for (row_idx, row) in spectrogram.fft_magnitude.axis_iter(Axis(0)).enumerate() {
            let energy = if let Some(values) = row.as_slice() {
                crate::simd::sum_squares_scaled(values, 1.0 / opts.nfft as f64)
            } else {
                row.iter().map(|v| v * v / opts.nfft as f64).sum::<f64>()
            };
            ceps[(row_idx, 0)] = replace_zero(energy).ln();
        }
    }

    apply_post_processing(ceps, opts)
}

pub(crate) fn spectrogram_with_fbanks(
    sig: &[f64],
    opts: &FeatureOptions,
    fbanks: &Matrix,
) -> Result<SpectrogramOutput> {
    let signal = if opts.pre_emph {
        crate::utils::preprocessing::pre_emphasis(sig, opts.pre_emph_coeff)
    } else {
        sig.to_vec()
    };

    let frames = crate::utils::preprocessing::framing(
        &signal,
        opts.fs,
        opts.window.win_len,
        opts.window.win_hop,
    )?;
    let windows = crate::utils::preprocessing::windowing(&frames, opts.window.win_type);
    let fft_magnitude = crate::utils::spectral::fft_magnitude(&windows, opts.nfft);
    let features = weighted_power_projection(&fft_magnitude, fbanks, 1.0 / opts.nfft as f64);
    Ok(SpectrogramOutput {
        features,
        fft_magnitude,
    })
}

pub(crate) fn weighted_power_projection(magnitude: &Matrix, fbanks: &Matrix, scale: f64) -> Matrix {
    #[cfg(not(feature = "portable-simd"))]
    {
        let power = magnitude.mapv(|value| value * value * scale);
        power.dot(&fbanks.t())
    }

    #[cfg(feature = "portable-simd")]
    {
        let mut out = Array2::<f64>::zeros((magnitude.nrows(), fbanks.nrows()));
        for (r, mag_row) in magnitude.axis_iter(Axis(0)).enumerate() {
            for (c, filter_row) in fbanks.axis_iter(Axis(0)).enumerate() {
                out[(r, c)] = match (mag_row.as_slice(), filter_row.as_slice()) {
                    (Some(mag), Some(filter)) => {
                        crate::simd::dot_square_weighted(mag, filter, scale)
                    }
                    _ => {
                        mag_row
                            .iter()
                            .zip(filter_row.iter())
                            .map(|(value, weight)| value * value * weight)
                            .sum::<f64>()
                            * scale
                    }
                };
            }
        }
        out
    }
}

impl From<&FeatureOptions> for FilterBankOptions {
    fn from(opts: &FeatureOptions) -> Self {
        Self {
            nfilts: opts.nfilts,
            nfft: opts.nfft,
            fs: opts.fs,
            low_freq: opts.low_freq,
            high_freq: opts.high_freq,
            scale: opts.scale,
        }
    }
}
