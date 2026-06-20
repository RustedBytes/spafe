use crate::{Matrix, Result, SpafeError};
use ndarray::Array2;
use num_complex::Complex64;
use rustfft::FftPlanner;

/// Frequency-domain ERB cosine filter bank and its frequency metadata.
#[derive(Debug, Clone)]
pub struct ErbCosFilterBank {
    /// Filter weights, one row per cochlear filter and one column per FFT bin.
    pub filters: Matrix,
    /// Center frequencies for lowpass, bandpass, and highpass filters in Hertz.
    pub center_freqs: Vec<f64>,
    /// Frequency grid in Hertz used by the filter-bank columns.
    pub freqs: Vec<f64>,
    /// Signal length, including any configured zero padding.
    pub padded_signal_size: usize,
}

/// Options for half-cosine filters spaced on the ERB scale.
#[derive(Debug, Clone)]
pub struct ErbCosFilterOptions {
    /// Number of filters used to span the frequency range at standard sampling.
    pub n: usize,
    /// Lower frequency limit in Hertz.
    pub low_lim: f64,
    /// Upper frequency limit in Hertz.
    pub high_lim: f64,
    /// ERB oversampling factor.
    pub sample_factor: usize,
    /// Whether to return a full FFT filter, including negative-frequency bins.
    pub full_filter: bool,
    /// Drop lowpass filters from the generated filter bank.
    pub no_lowpass: bool,
    /// Drop highpass filters from the generated filter bank.
    pub no_highpass: bool,
    /// Reject odd sample factors and high limits above Nyquist.
    pub strict: bool,
}

impl Default for ErbCosFilterOptions {
    fn default() -> Self {
        Self {
            n: 40,
            low_lim: 20.0,
            high_lim: 10_000.0,
            sample_factor: 4,
            full_filter: false,
            no_lowpass: false,
            no_highpass: false,
            strict: true,
        }
    }
}

/// Envelope extraction applied after cochlear subband filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvelopeMode {
    /// Build an analytic signal from positive-frequency bins and return its magnitude.
    Hilbert,
    /// Inverse-transform subbands and return absolute values.
    AbsSubbands,
    /// Inverse-transform subbands and half-wave rectify them.
    RectifySubbands,
}

/// Downsampling method for cochlear envelopes.
#[derive(Debug, Clone)]
pub enum DownsamplingMode {
    /// Sinc low-pass filter windowed by a Kaiser window.
    SincWithKaiserWindow {
        /// FIR window size.
        window_size: usize,
        /// Optional explicit left/right zero padding before convolution.
        padding: Option<(usize, usize)>,
    },
    /// Weighted average pooling with a Hann window.
    HannPooling1d {
        /// FIR window size.
        window_size: usize,
        /// Symmetric zero padding before convolution.
        padding: usize,
        /// Normalize the Hann weights to unit sum.
        normalize: bool,
    },
}

impl Default for DownsamplingMode {
    fn default() -> Self {
        Self::SincWithKaiserWindow {
            window_size: 1001,
            padding: None,
        }
    }
}

/// Elementwise compression used after envelope extraction and downsampling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompressionMode {
    /// Apply only scale and offset after rectification.
    Linear {
        /// Multiplicative scale.
        scale: f64,
        /// Additive offset before scaling.
        offset: f64,
    },
    /// Raise rectified, scaled values to a power.
    Power {
        /// Multiplicative scale.
        scale: f64,
        /// Additive offset before scaling.
        offset: f64,
        /// Compression exponent.
        power: f64,
    },
    /// Forward-pass equivalent of chcochleagram's clipped-gradient power compression.
    ClippedGradPower {
        /// Multiplicative scale.
        scale: f64,
        /// Additive offset before scaling.
        offset: f64,
        /// Compression exponent.
        power: f64,
        /// Gradient clipping value in the Python/autograd implementation.
        clip_value: f64,
    },
}

impl Default for CompressionMode {
    fn default() -> Self {
        Self::ClippedGradPower {
            scale: 1.0,
            offset: 1e-8,
            power: 0.3,
            clip_value: 5.0,
        }
    }
}

/// Options for audio-to-cochleagram conversion.
#[derive(Debug, Clone)]
pub struct CochleagramOptions {
    /// Expected input signal size.
    pub signal_size: usize,
    /// Input sampling rate in Hertz.
    pub sr: usize,
    /// Envelope sampling rate after downsampling.
    pub env_sr: usize,
    /// Optional zero-padding factor applied before filtering.
    pub pad_factor: Option<usize>,
    /// Use real-FFT positive-frequency filters.
    pub use_rfft: bool,
    /// ERB cosine filter configuration.
    pub filter: ErbCosFilterOptions,
    /// Envelope extraction method.
    pub envelope: EnvelopeMode,
    /// Downsampling method.
    pub downsampling: DownsamplingMode,
    /// Optional output compression.
    pub compression: Option<CompressionMode>,
    /// Apply downsampling before compression when true.
    pub downsample_then_compress: bool,
}

impl Default for CochleagramOptions {
    fn default() -> Self {
        Self {
            signal_size: 40_000,
            sr: 20_000,
            env_sr: 200,
            pad_factor: None,
            use_rfft: true,
            filter: ErbCosFilterOptions {
                n: 50,
                low_lim: 50.0,
                high_lim: 10_000.0,
                sample_factor: 4,
                full_filter: false,
                ..ErbCosFilterOptions::default()
            },
            envelope: EnvelopeMode::Hilbert,
            downsampling: DownsamplingMode::default(),
            compression: Some(CompressionMode::default()),
            downsample_then_compress: true,
        }
    }
}

/// Intermediate arrays from cochleagram generation.
#[derive(Debug, Clone)]
pub struct CochleagramLatents {
    /// Complex frequency-domain subbands, indexed as `[filter][frequency_bin]`.
    pub subbands: Vec<Vec<Complex64>>,
    /// Time-domain envelopes before downsampling.
    pub envelopes: Matrix,
    /// Envelopes after downsampling and before optional compression when
    /// `downsample_then_compress` is true.
    pub downsampled: Matrix,
}

/// Cochleagram output and optional intermediate arrays.
#[derive(Debug, Clone)]
pub struct CochleagramOutput {
    /// Cochleagram matrix, one row per cochlear filter.
    pub cochleagram: Matrix,
    /// Intermediate arrays useful for debugging and parity checks.
    pub latents: CochleagramLatents,
    /// Filter bank used to produce the cochleagram.
    pub filter_bank: ErbCosFilterBank,
}

/// Convert Hertz to ERB using the Glasberg-Moore formula used by chcochleagram.
pub fn freq2erb(freq_hz: f64) -> f64 {
    9.265 * (1.0 + freq_hz / (24.7 * 9.265)).ln()
}

/// Convert ERB to Hertz using the Glasberg-Moore formula used by chcochleagram.
pub fn erb2freq(n_erb: f64) -> f64 {
    24.7 * 9.265 * ((n_erb / 9.265).exp() - 1.0)
}

/// Generate a half-cosine filter over the values within `(low, high)`.
pub fn make_cosine_filter(freqs: &[f64], low: f64, high: f64, convert_to_erb: bool) -> Vec<f64> {
    let (freqs_erb, low_erb, high_erb) = if convert_to_erb {
        (
            freqs.iter().map(|freq| freq2erb(*freq)).collect::<Vec<_>>(),
            freq2erb(low),
            freq2erb(high),
        )
    } else {
        (freqs.to_vec(), low, high)
    };
    cosine_values_in_erb(&freqs_erb, low_erb, high_erb)
}

/// Create ERB-spaced half-cosine filters for cochleagram generation.
pub fn erb_cos_filter_bank(
    signal_size: usize,
    sr: usize,
    pad_factor: Option<usize>,
    opts: &ErbCosFilterOptions,
) -> Result<ErbCosFilterBank> {
    if opts.sample_factor == 0 {
        return Err(SpafeError::InvalidParameter(
            "sample_factor must be positive",
        ));
    }
    if opts.sample_factor != 1 && !opts.sample_factor.is_multiple_of(2) && opts.strict {
        return Err(SpafeError::InvalidParameter(
            "sample_factor must be one or even",
        ));
    }

    let padded_signal_size = pad_factor.unwrap_or(1).max(1) * signal_size;
    let (n_freqs, max_freq) = if padded_signal_size.is_multiple_of(2) {
        (padded_signal_size / 2, sr as f64 / 2.0)
    } else {
        (
            (padded_signal_size - 1) / 2,
            sr as f64 * (padded_signal_size - 1) as f64 / 2.0 / padded_signal_size as f64,
        )
    };
    let high_lim = if opts.high_lim > sr as f64 / 2.0 {
        if opts.strict {
            return Err(SpafeError::HighFrequency);
        }
        max_freq
    } else {
        opts.high_lim
    };

    let n_filters = opts.sample_factor * (opts.n + 1) - 1;
    let n_lp_hp = 2 * opts.sample_factor;
    let freqs = linspace_closed(0.0, max_freq, n_freqs + 1);
    let mut filts = Array2::<f64>::zeros((n_freqs + 1, n_filters + n_lp_hp));
    let low_erb = freq2erb(opts.low_lim);
    let high_erb = freq2erb(high_lim);
    let erb_spacing = (high_erb - low_erb) / (n_filters + 1) as f64;
    let center_freqs_erb = (1..=n_filters)
        .map(|idx| low_erb + erb_spacing * idx as f64)
        .collect::<Vec<_>>();
    let freqs_erb = freqs.iter().map(|freq| freq2erb(*freq)).collect::<Vec<_>>();

    for (idx, center) in center_freqs_erb.iter().enumerate() {
        let col = idx + opts.sample_factor;
        let low = center - opts.sample_factor as f64 * erb_spacing;
        let high = center + opts.sample_factor as f64 * erb_spacing;
        let avg = (low + high) / 2.0;
        let range = high - low;
        for (row, freq_erb) in freqs_erb.iter().enumerate() {
            if *freq_erb > low && *freq_erb < high {
                filts[(row, col)] = ((*freq_erb - avg) / range * std::f64::consts::PI).cos();
            }
        }
    }

    for idx in 0..opts.sample_factor {
        let offset = idx + opts.sample_factor;
        let lowpass_peak = erb2freq(center_freqs_erb[idx]);
        let lp_h_ind = freqs
            .iter()
            .rposition(|freq| *freq < lowpass_peak)
            .ok_or(SpafeError::InvalidParameter("invalid lowpass filter peak"))?;
        for row in 0..=lp_h_ind {
            filts[(row, idx)] = (1.0 - filts[(row, offset)].powi(2)).max(0.0).sqrt();
        }

        let highpass_peak = erb2freq(center_freqs_erb[n_filters - 1 - idx]);
        let hp_l_ind = freqs
            .iter()
            .position(|freq| *freq > highpass_peak)
            .ok_or(SpafeError::InvalidParameter("invalid highpass filter peak"))?;
        let hp_col = filts.ncols() - 1 - idx;
        let source_col = filts.ncols() - 1 - offset;
        for row in hp_l_ind..filts.nrows() {
            filts[(row, hp_col)] = (1.0 - filts[(row, source_col)].powi(2)).max(0.0).sqrt();
        }
    }

    let scale = (opts.sample_factor as f64).sqrt();
    filts.mapv_inplace(|value| value / scale);

    let mut center_freqs = center_freqs_erb[..opts.sample_factor]
        .iter()
        .map(|center| center - opts.sample_factor as f64 * erb_spacing)
        .chain(center_freqs_erb.iter().copied())
        .chain(
            center_freqs_erb[n_filters - opts.sample_factor..]
                .iter()
                .map(|center| center + opts.sample_factor as f64 * erb_spacing),
        )
        .map(erb2freq)
        .map(|freq| if freq < 0.0 { 1.0 } else { freq })
        .collect::<Vec<_>>();

    if opts.no_lowpass {
        filts = filts
            .slice(ndarray::s![.., opts.sample_factor..])
            .to_owned();
        center_freqs.drain(..opts.sample_factor);
    }
    if opts.no_highpass {
        let keep = filts.ncols().saturating_sub(opts.sample_factor);
        filts = filts.slice(ndarray::s![.., ..keep]).to_owned();
        center_freqs.truncate(keep);
    }

    let filters = if opts.full_filter {
        make_full_filter_set(&filts, padded_signal_size)
    } else {
        filts.t().to_owned()
    };

    Ok(ErbCosFilterBank {
        filters,
        center_freqs,
        freqs,
        padded_signal_size,
    })
}

/// Compute complex frequency-domain subbands by multiplying an RFFT with filters.
pub fn compute_subbands(
    sig: &[f64],
    filter_bank: &ErbCosFilterBank,
) -> Result<Vec<Vec<Complex64>>> {
    if sig.len() > filter_bank.padded_signal_size {
        return Err(SpafeError::InvalidParameter(
            "signal is longer than the configured cochleagram size",
        ));
    }
    let spectrum = rfft_padded(sig, filter_bank.padded_signal_size);
    if spectrum.len() != filter_bank.filters.ncols() {
        return Err(SpafeError::InvalidParameter(
            "filter width does not match RFFT size",
        ));
    }

    Ok(filter_bank
        .filters
        .rows()
        .into_iter()
        .map(|row| {
            row.iter()
                .zip(spectrum.iter())
                .map(|(weight, bin)| *bin * *weight)
                .collect::<Vec<_>>()
        })
        .collect())
}

/// Extract time-domain envelopes from complex subbands.
pub fn extract_envelopes(
    subbands: &[Vec<Complex64>],
    signal_size: usize,
    padded_signal_size: usize,
    mode: EnvelopeMode,
) -> Matrix {
    let mut out = Array2::<f64>::zeros((subbands.len(), signal_size));
    for (row, subband) in subbands.iter().enumerate() {
        let values = match mode {
            EnvelopeMode::Hilbert => hilbert_envelope(subband, padded_signal_size),
            EnvelopeMode::AbsSubbands => irfft_time(subband, padded_signal_size)
                .into_iter()
                .map(f64::abs)
                .collect(),
            EnvelopeMode::RectifySubbands => irfft_time(subband, padded_signal_size)
                .into_iter()
                .map(|value| value.max(0.0))
                .collect(),
        };
        for (col, value) in values.into_iter().take(signal_size).enumerate() {
            out[(row, col)] = value;
        }
    }
    out
}

/// Downsample cochlear envelopes.
pub fn downsample_envelopes(
    envelopes: &Matrix,
    sr: usize,
    env_sr: usize,
    mode: &DownsamplingMode,
) -> Result<Matrix> {
    if env_sr == 0 || !sr.is_multiple_of(env_sr) {
        return Err(SpafeError::InvalidParameter(
            "downsampling requires an integer factor",
        ));
    }
    let stride = sr / env_sr;
    let (kernel, padding) = match mode {
        DownsamplingMode::SincWithKaiserWindow {
            window_size,
            padding,
        } => (
            sinc_with_kaiser(*window_size, stride),
            padding.unwrap_or((0, 0)),
        ),
        DownsamplingMode::HannPooling1d {
            window_size,
            padding,
            normalize,
        } => (hann_window(*window_size, *normalize), (*padding, *padding)),
    };
    Ok(convolve_rows_valid(envelopes, &kernel, stride, padding))
}

/// Apply elementwise cochleagram compression.
pub fn apply_compression(x: &Matrix, compression: CompressionMode) -> Matrix {
    match compression {
        CompressionMode::Linear { scale, offset } => {
            x.mapv(|value| scale * (value.max(0.0) + offset))
        }
        CompressionMode::Power {
            scale,
            offset,
            power,
        }
        | CompressionMode::ClippedGradPower {
            scale,
            offset,
            power,
            ..
        } => x.mapv(|value| (scale * (value.max(0.0) + offset)).powf(power)),
    }
}

/// Compute a cochleagram for one audio signal.
pub fn cochleagram(sig: &[f64], opts: &CochleagramOptions) -> Result<CochleagramOutput> {
    if !opts.use_rfft {
        return Err(SpafeError::InvalidParameter(
            "only use_rfft=true is supported",
        ));
    }
    if sig.len() != opts.signal_size {
        return Err(SpafeError::InvalidParameter(
            "signal length must match opts.signal_size",
        ));
    }

    let filter_bank =
        erb_cos_filter_bank(opts.signal_size, opts.sr, opts.pad_factor, &opts.filter)?;
    let subbands = compute_subbands(sig, &filter_bank)?;
    let envelopes = extract_envelopes(
        &subbands,
        opts.signal_size,
        filter_bank.padded_signal_size,
        opts.envelope,
    );

    let (downsampled, cochleagram) = if opts.downsample_then_compress {
        let downsampled =
            downsample_envelopes(&envelopes, opts.sr, opts.env_sr, &opts.downsampling)?;
        let cochleagram = if let Some(compression) = opts.compression {
            apply_compression(&downsampled, compression)
        } else {
            downsampled.clone()
        };
        (downsampled, cochleagram)
    } else {
        let compressed = if let Some(compression) = opts.compression {
            apply_compression(&envelopes, compression)
        } else {
            envelopes.clone()
        };
        let downsampled =
            downsample_envelopes(&compressed, opts.sr, opts.env_sr, &opts.downsampling)?;
        (downsampled.clone(), downsampled)
    };

    Ok(CochleagramOutput {
        cochleagram,
        latents: CochleagramLatents {
            subbands,
            envelopes,
            downsampled,
        },
        filter_bank,
    })
}

/// Return the Rust equivalent of chcochleagram's `cochleagram_1` preset.
pub fn cochleagram_1_options() -> CochleagramOptions {
    CochleagramOptions::default()
}

/// Calculate TensorFlow-style "same" padding for a one-dimensional convolution.
pub fn calculate_same_padding(
    input_size: usize,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
) -> (usize, usize) {
    let pad = ((input_size.div_ceil(stride) - 1) * stride + kernel_size - input_size) * dilation;
    (pad / 2, pad - pad / 2)
}

fn cosine_values_in_erb(freqs_erb: &[f64], low_erb: f64, high_erb: f64) -> Vec<f64> {
    let avg = (low_erb + high_erb) / 2.0;
    let range = high_erb - low_erb;
    freqs_erb
        .iter()
        .filter(|freq| **freq > low_erb && **freq < high_erb)
        .map(|freq| ((*freq - avg) / range * std::f64::consts::PI).cos())
        .collect()
}

fn linspace_closed(start: f64, end: f64, n: usize) -> Vec<f64> {
    match n {
        0 => Vec::new(),
        1 => vec![start],
        _ => {
            let step = (end - start) / (n - 1) as f64;
            (0..n).map(|idx| start + idx as f64 * step).collect()
        }
    }
}

fn make_full_filter_set(filts: &Matrix, signal_length: usize) -> Matrix {
    let neg_start = 1;
    let neg_end = if signal_length.is_multiple_of(2) {
        filts.nrows().saturating_sub(1)
    } else {
        filts.nrows()
    };
    let full_rows = filts.nrows() + neg_end.saturating_sub(neg_start);
    let mut full = Array2::<f64>::zeros((full_rows, filts.ncols()));
    full.slice_mut(ndarray::s![..filts.nrows(), ..])
        .assign(filts);
    for (dst, src) in (neg_start..neg_end).rev().enumerate() {
        full.row_mut(filts.nrows() + dst).assign(&filts.row(src));
    }
    full.t().to_owned()
}

fn rfft_padded(sig: &[f64], nfft: usize) -> Vec<Complex64> {
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(nfft);
    let mut buffer = vec![Complex64::new(0.0, 0.0); nfft];
    for (dst, src) in buffer.iter_mut().zip(sig.iter()) {
        dst.re = *src;
    }
    fft.process(&mut buffer);
    buffer[..(nfft / 2 + 1)].to_vec()
}

fn hilbert_envelope(subband: &[Complex64], signal_size: usize) -> Vec<f64> {
    let mut planner = FftPlanner::<f64>::new();
    let ifft = planner.plan_fft_inverse(signal_size);
    let mut buffer = vec![Complex64::new(0.0, 0.0); signal_size];
    for (dst, src) in buffer.iter_mut().zip(subband.iter()) {
        *dst = *src;
    }
    ifft.process(&mut buffer);
    buffer
        .into_iter()
        .map(|value| (value / signal_size as f64).norm().max(1e-8))
        .collect()
}

fn irfft_time(subband: &[Complex64], signal_size: usize) -> Vec<f64> {
    let mut planner = FftPlanner::<f64>::new();
    let ifft = planner.plan_fft_inverse(signal_size);
    let mut buffer = vec![Complex64::new(0.0, 0.0); signal_size];
    for (idx, value) in subband.iter().enumerate() {
        buffer[idx] = *value;
    }
    let last_mirror = if signal_size.is_multiple_of(2) {
        subband.len().saturating_sub(1)
    } else {
        subband.len()
    };
    for idx in 1..last_mirror {
        buffer[signal_size - idx] = subband[idx].conj();
    }
    ifft.process(&mut buffer);
    buffer
        .into_iter()
        .map(|value| (value / signal_size as f64).re)
        .collect()
}

fn sinc_with_kaiser(window_size: usize, downsample_factor: usize) -> Vec<f64> {
    let window = kaiser_window(window_size, 5.0);
    (0..window_size)
        .map(|idx| {
            let time = -(window_size as f64) / 2.0 + idx as f64;
            window[idx] * sinc(time / downsample_factor as f64) / downsample_factor as f64
        })
        .collect()
}

fn hann_window(window_size: usize, normalize: bool) -> Vec<f64> {
    let mut window = if window_size <= 1 {
        vec![1.0; window_size]
    } else {
        (0..window_size)
            .map(|idx| {
                0.5 - 0.5
                    * (2.0 * std::f64::consts::PI * idx as f64 / (window_size - 1) as f64).cos()
            })
            .collect::<Vec<_>>()
    };
    if normalize {
        let sum = window.iter().sum::<f64>();
        if sum != 0.0 {
            for value in &mut window {
                *value /= sum;
            }
        }
    }
    window
}

fn kaiser_window(window_size: usize, beta: f64) -> Vec<f64> {
    if window_size <= 1 {
        return vec![1.0; window_size];
    }
    let denom = bessel_i0(beta);
    (0..window_size)
        .map(|idx| {
            let ratio = 2.0 * idx as f64 / (window_size - 1) as f64 - 1.0;
            bessel_i0(beta * (1.0 - ratio * ratio).max(0.0).sqrt()) / denom
        })
        .collect()
}

fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    for k in 1..40 {
        term *= (x * x / 4.0) / (k * k) as f64;
        sum += term;
    }
    sum
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let pix = std::f64::consts::PI * x;
        pix.sin() / pix
    }
}

fn convolve_rows_valid(
    x: &Matrix,
    kernel: &[f64],
    stride: usize,
    padding: (usize, usize),
) -> Matrix {
    let input_cols = x.ncols() + padding.0 + padding.1;
    if input_cols < kernel.len() || stride == 0 {
        return Array2::<f64>::zeros((x.nrows(), 0));
    }
    let out_cols = (input_cols - kernel.len()) / stride + 1;
    let mut out = Array2::<f64>::zeros((x.nrows(), out_cols));
    for row in 0..x.nrows() {
        let mut padded = vec![0.0; input_cols];
        for col in 0..x.ncols() {
            padded[padding.0 + col] = x[(row, col)];
        }
        for out_col in 0..out_cols {
            let start = out_col * stride;
            out[(row, out_col)] = kernel
                .iter()
                .enumerate()
                .map(|(idx, coeff)| padded[start + idx] * coeff)
                .sum();
        }
    }
    out
}
