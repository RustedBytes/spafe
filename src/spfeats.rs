use num_complex::Complex64;
use rustfft::FftPlanner;

/// Collection of spectral descriptors returned by [`extract_feats`] and [`extract_feats_with_nfft`].
#[derive(Debug, Clone, Default)]
pub struct SpectralFeats {
    /// Spectrum barycenter.
    pub spectral_centroid: f64,
    /// Asymmetry of the spectrum around its centroid.
    pub spectral_skewness: f64,
    /// Flatness/peakedness of the spectrum around its centroid.
    pub spectral_kurtosis: f64,
    /// Entropy of the magnitude spectrum.
    pub spectral_entropy: f64,
    /// Spread of the spectrum around the centroid.
    pub spectral_spread: Vec<f64>,
    /// Geometric-to-arithmetic mean ratio of the magnitude spectrum.
    pub spectral_flatness: f64,
    /// Frequency-scaled spectrum values matching the Python descriptor output.
    pub spectral_rolloff: Vec<Complex64>,
    /// Frame-to-frame magnitude-spectrum change.
    pub spectral_flux: f64,
    /// Mean complex FFT bin value.
    pub spectral_mean: Complex64,
    /// Root-mean-square complex FFT value.
    pub spectral_rms: Complex64,
    /// Standard deviation of complex FFT bins.
    pub spectral_std: f64,
    /// Variance of complex FFT bins.
    pub spectral_variance: f64,
}

/// Compute the spectral centroid, the barycenter of the spectrum.
pub fn spectral_centroid(fs: usize, spectrum: &[Complex64], order: usize) -> f64 {
    let magnitude = magnitude_spectrum(spectrum);
    let freqs = fftfreq_abs(spectrum.len(), fs);
    let denom = magnitude.iter().sum::<f64>();
    magnitude
        .iter()
        .zip(freqs.iter())
        .map(|(mag, freq)| mag * freq.powi(order as i32))
        .sum::<f64>()
        / denom
}

/// Compute spectral skewness, a measure of asymmetry around the spectral centroid.
pub fn spectral_skewness(_sig: &[f64], fs: usize, spectrum: &[Complex64]) -> f64 {
    let magnitude = magnitude_spectrum(spectrum);
    let freqs = fftfreq_abs(spectrum.len(), fs);
    let mu1 = spectral_centroid(fs, spectrum, 1);
    let mu2 = spectral_centroid(fs, spectrum, 2);
    magnitude
        .iter()
        .zip(freqs.iter())
        .map(|(mag, freq)| mag * (freq - mu1).powi(3))
        .sum::<f64>()
        / (magnitude.iter().sum::<f64>() * mu2.powi(3))
}

/// Compute spectral kurtosis, a measure of spectral flatness around the centroid.
pub fn spectral_kurtosis(_sig: &[f64], fs: usize, spectrum: &[Complex64]) -> f64 {
    let magnitude = magnitude_spectrum(spectrum);
    let freqs = fftfreq_abs(spectrum.len(), fs);
    let mu1 = spectral_centroid(fs, spectrum, 1);
    let mu2 = spectral_centroid(fs, spectrum, 2);
    magnitude
        .iter()
        .zip(freqs.iter())
        .map(|(mag, freq)| mag * (freq - mu1).powi(4))
        .sum::<f64>()
        / (magnitude.iter().sum::<f64>() * mu2.powi(4))
}

/// Compute spectral entropy from a magnitude spectrum.
pub fn spectral_entropy(spectrum: &[Complex64]) -> f64 {
    let magnitude = magnitude_spectrum(spectrum);
    -magnitude.iter().map(|mag| mag * mag.ln()).sum::<f64>() / (magnitude.len() as f64).ln()
}

/// Compute spectral spread around the spectral centroid.
pub fn spectral_spread(_sig: &[f64], fs: usize, spectrum: &[Complex64]) -> Vec<f64> {
    let magnitude = magnitude_spectrum(spectrum);
    let freqs = fftfreq_abs(spectrum.len(), fs);
    let mu1 = spectral_centroid(fs, spectrum, 1);
    let sum_delta = freqs.iter().map(|freq| freq - mu1).sum::<f64>();
    let denom = magnitude.iter().sum::<f64>();
    magnitude
        .iter()
        .map(|mag| ((sum_delta.powi(2) * mag) / denom).sqrt())
        .collect()
}

/// Compute spectral flatness.
pub fn spectral_flatness(spectrum: &[Complex64]) -> f64 {
    let magnitude = magnitude_spectrum(spectrum);
    if magnitude.contains(&0.0) {
        return 0.0;
    }
    let gmean =
        (magnitude.iter().map(|value| value.ln()).sum::<f64>() / magnitude.len() as f64).exp();
    gmean / (magnitude.iter().sum::<f64>() / magnitude.len() as f64)
}

/// Compute the spectral roll-off energy threshold for percentage `k`.
pub fn spectral_rolloff(spectrum: &[Complex64], k: f64) -> f64 {
    k * magnitude_spectrum(spectrum).iter().sum::<f64>()
}

/// Compute spectral flux with a `p`-norm.
pub fn spectral_flux(spectrum: &[Complex64], p: usize) -> f64 {
    let magnitude = magnitude_spectrum(spectrum);
    magnitude
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).abs().powi(p as i32))
        .sum::<f64>()
        .powf(1.0 / p as f64)
}

/// Compute spectral descriptors with the default `nfft` of 512.
pub fn extract_feats(sig: &[f64], fs: usize) -> SpectralFeats {
    extract_feats_with_nfft(sig, fs, 512)
}

/// Compute spectral descriptors for an audio signal.
pub fn extract_feats_with_nfft(sig: &[f64], fs: usize, nfft: usize) -> SpectralFeats {
    let spectrum = rfft(sig, nfft);
    let squared = spectrum
        .iter()
        .map(|value| *value * *value)
        .collect::<Vec<_>>();
    SpectralFeats {
        spectral_centroid: spectral_centroid(fs, &spectrum, 1),
        spectral_skewness: spectral_skewness(sig, fs, &spectrum),
        spectral_kurtosis: spectral_kurtosis(sig, fs, &spectrum),
        spectral_entropy: spectral_entropy(&spectrum),
        spectral_spread: spectral_spread(sig, fs, &spectrum),
        spectral_flatness: spectral_flatness(&spectrum),
        spectral_rolloff: spectrum.iter().map(|value| *value * fs as f64).collect(),
        spectral_flux: spectral_flux(&spectrum, 2),
        spectral_mean: complex_mean(&spectrum),
        spectral_rms: complex_mean(&squared).sqrt(),
        spectral_std: complex_variance(&spectrum).sqrt(),
        spectral_variance: complex_variance(&spectrum),
    }
}

fn rfft(sig: &[f64], nfft: usize) -> Vec<Complex64> {
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(nfft);
    let mut buffer = vec![Complex64::new(0.0, 0.0); nfft];
    for (dst, src) in buffer.iter_mut().zip(sig.iter()) {
        dst.re = *src;
    }
    fft.process(&mut buffer);
    buffer[..(nfft / 2 + 1)].to_vec()
}

fn magnitude_spectrum(spectrum: &[Complex64]) -> Vec<f64> {
    spectrum.iter().map(|value| value.norm()).collect()
}

fn fftfreq_abs(n: usize, fs: usize) -> Vec<f64> {
    let step = fs as f64 / n as f64;
    (0..n)
        .map(|i| {
            let value = if i <= (n - 1) / 2 {
                i as f64
            } else {
                i as f64 - n as f64
            };
            (value * step).abs()
        })
        .collect()
}

fn complex_mean(values: &[Complex64]) -> Complex64 {
    values.iter().sum::<Complex64>() / values.len() as f64
}

fn complex_variance(values: &[Complex64]) -> f64 {
    let mean = complex_mean(values);
    values
        .iter()
        .map(|value| (*value - mean).norm_sqr())
        .sum::<f64>()
        / values.len() as f64
}
