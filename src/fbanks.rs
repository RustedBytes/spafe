use crate::utils::converters::*;
use crate::utils::filters::scale_fbank;
use crate::*;
use ndarray::{Array2, Axis};
use num_complex::Complex64;

/// Compute Mel filter banks.
///
/// The returned matrix stores one filter per row and FFT bins in columns. Center frequencies are
/// returned in the selected Mel scale.
pub fn mel_filter_banks(
    opts: &FilterBankOptions,
    conversion: MelConversionApproach,
) -> Result<(Matrix, Vector)> {
    mel_filter_banks_helper(opts, "mel", conversion)
}

/// Compute inverse Mel filter banks.
///
/// This mirrors the Python inverse-MFCC filter-bank helper: scaling is reversed and each filter row
/// is mirrored while preserving the Mel-scale center frequencies.
pub fn inverse_mel_filter_banks(
    opts: &FilterBankOptions,
    conversion: MelConversionApproach,
) -> Result<(Matrix, Vector)> {
    let scale = match opts.scale {
        Scale::Ascendant => Scale::Descendant,
        Scale::Descendant => Scale::Ascendant,
        Scale::Constant => Scale::Constant,
    };
    let mut inv_opts = opts.clone();
    inv_opts.scale = scale;
    let (mut fbank, mel_freqs) = mel_filter_banks_helper(&inv_opts, "mel", conversion)?;
    for mut row in fbank.axis_iter_mut(Axis(0)) {
        let reversed: Vec<f64> = row.iter().rev().copied().collect();
        for (dst, src) in row.iter_mut().zip(reversed) {
            *dst = src.abs();
        }
    }
    let high = checked_high_freq(opts.low_freq, opts.high_freq, opts.fs)?;
    let center_freqs = mel_freqs.mapv(|freq| hz2mel(high - mel2hz(freq, conversion), conversion));
    Ok((fbank, center_freqs))
}

/// Compute linear-frequency filter banks.
///
/// The returned matrix stores one filter per row and FFT bins in columns.
pub fn linear_filter_banks(opts: &FilterBankOptions) -> Result<(Matrix, Vector)> {
    let (fbank, freqs) = mel_filter_banks_helper(opts, "lin", MelConversionApproach::Oshaghnessy)?;
    Ok((fbank.mapv(f64::abs), freqs))
}

/// Shared triangular filter-bank builder used by Mel and linear filter banks.
///
/// Pass `"mel"` for Mel-spaced filters or `"lin"` for linearly spaced filters. The returned vector
/// contains the center frequencies in the corresponding scale.
pub fn mel_filter_banks_helper(
    opts: &FilterBankOptions,
    fb_type: &str,
    conversion: MelConversionApproach,
) -> Result<(Matrix, Vector)> {
    let high_freq = checked_high_freq(opts.low_freq, opts.high_freq, opts.fs)?;
    let nfilts = opts.nfilts;
    let (lower_hz, center_hz, upper_hz, center_freqs) = if fb_type == "mel" {
        let low_mel = hz2mel(opts.low_freq, conversion);
        let high_mel = hz2mel(high_freq, conversion);
        let delta = (high_mel - low_mel).abs() / (nfilts + 1) as f64;
        let scale_freqs: Vec<f64> = (0..nfilts + 2)
            .map(|i| low_mel + delta * i as f64)
            .collect();
        let lower_hz: Vec<f64> = scale_freqs[..nfilts]
            .iter()
            .map(|v| mel2hz(*v, conversion))
            .collect();
        let center_hz: Vec<f64> = scale_freqs[1..nfilts + 1]
            .iter()
            .map(|v| mel2hz(*v, conversion))
            .collect();
        let upper_hz: Vec<f64> = scale_freqs[2..]
            .iter()
            .map(|v| mel2hz(*v, conversion))
            .collect();
        (
            lower_hz,
            center_hz,
            upper_hz,
            Vector::from_vec(scale_freqs[1..nfilts + 1].to_vec()),
        )
    } else {
        let delta = (high_freq - opts.low_freq).abs() / (nfilts + 1) as f64;
        let scale_freqs: Vec<f64> = (0..nfilts + 2)
            .map(|i| opts.low_freq + delta * i as f64)
            .collect();
        (
            scale_freqs[..nfilts].to_vec(),
            scale_freqs[1..nfilts + 1].to_vec(),
            scale_freqs[2..].to_vec(),
            Vector::from_vec(scale_freqs[1..nfilts + 1].to_vec()),
        )
    };

    let freqs = linspace(opts.low_freq, high_freq, opts.nfft / 2 + 1);
    let mut fbank = Array2::<f64>::zeros((nfilts, opts.nfft / 2 + 1));
    for j in 0..nfilts {
        let lower = lower_hz[j];
        let center = center_hz[j];
        let upper = upper_hz[j];
        for (i, freq) in freqs.iter().enumerate() {
            if *freq >= lower && *freq <= center {
                fbank[(j, i)] = (*freq - lower) / (center - lower);
            }
            if *freq >= center && *freq <= upper {
                fbank[(j, i)] = (upper - *freq) / (upper - center);
            }
        }
    }
    apply_scaling(fbank, opts.scale).map(|scaled| (scaled, center_freqs))
}

/// Compute a Bark filter response around a Bark-scale center frequency.
pub fn bark_fm(fb: f64, fc: f64) -> f64 {
    let diff = fb - fc;
    if !(-1.3..=2.5).contains(&diff) {
        0.0
    } else if (-1.3..=-0.5).contains(&diff) {
        10.0_f64.powf(2.5 * (diff + 0.5))
    } else if diff < 0.5 {
        1.0
    } else {
        10.0_f64.powf(-(diff - 0.5))
    }
}

/// Compatibility alias for the Python `Fm` Bark filter response helper.
pub fn fm(fb: f64, fc: f64) -> f64 {
    bark_fm(fb, fc)
}

/// Compute Bark filter banks.
///
/// The returned matrix stores one filter per row and FFT bins in columns. Center frequencies are
/// returned in Bark units for the selected conversion approach.
pub fn bark_filter_banks(
    opts: &FilterBankOptions,
    conversion: BarkConversionApproach,
) -> Result<(Matrix, Vector)> {
    let high_freq = checked_high_freq(opts.low_freq, opts.high_freq, opts.fs)?;
    let low_bark = hz2bark(opts.low_freq, conversion);
    let high_bark = hz2bark(high_freq, conversion);
    let centers = linspace(low_bark, high_bark, opts.nfilts);
    let bins: Vec<usize> = centers
        .iter()
        .map(|f| {
            ((opts.nfft + 1) as f64 * (bark2hz(*f, conversion) / opts.fs as f64)).floor() as usize
        })
        .collect();
    let mut fbank = Array2::<f64>::zeros((opts.nfilts, opts.nfft / 2 + 1));
    if let (Some(first), Some(last)) = (bins.first(), bins.last()) {
        for j in 0..opts.nfilts {
            for i in *first..(*last).min(opts.nfft / 2 + 1) {
                let fb = hz2bark(
                    (i as f64 * opts.fs as f64) / (opts.nfft + 1) as f64,
                    conversion,
                );
                fbank[(j, i)] = bark_fm(fb, centers[j]);
            }
        }
    }
    apply_scaling(fbank, opts.scale).map(|scaled| (scaled, Vector::from_vec(centers)))
}

const EAR_Q: f64 = 9.26449;
const MIN_BW: f64 = 24.7;

/// Compute gammatone center frequencies on the ERB scale.
pub fn generate_center_frequencies(min_freq: f64, max_freq: f64, nfilts: usize) -> Vector {
    let c = EAR_Q * MIN_BW;
    let mut centers: Vec<f64> = (1..=nfilts)
        .map(|m| {
            (max_freq + c)
                * ((m as f64 / nfilts as f64) * ((min_freq + c).ln() - (max_freq + c).ln())).exp()
                - c
        })
        .collect();
    centers.reverse();
    Vector::from_vec(centers)
}

/// Compute gammatone gain values and intermediate coefficients.
pub fn compute_gain(fcs: &[f64], b: &[f64], wt: &[f64], t: f64) -> (Matrix, Vector) {
    let n = fcs.len();
    let mut a = Array2::<f64>::zeros((4, n));
    let mut gain = Vec::with_capacity(n);
    let smax = (3.0 + 2.0_f64.powf(1.5)).sqrt();
    let smin = (3.0 - 2.0_f64.powf(1.5)).sqrt();

    for j in 0..n {
        let k = (b[j] * t).exp();
        let cos = (2.0 * fcs[j] * std::f64::consts::PI * t).cos();
        let sin = (2.0 * fcs[j] * std::f64::consts::PI * t).sin();
        a[(0, j)] = (cos + smax * sin) / k;
        a[(1, j)] = (cos - smax * sin) / k;
        a[(2, j)] = (cos + smin * sin) / k;
        a[(3, j)] = (cos - smin * sin) / k;

        let kj = Complex64::from_polar(1.0, wt[j]);
        let mut g = Complex64::new(1.0, 0.0);
        for row in 0..4 {
            g *= 2.0 * t * kj * (Complex64::new(a[(row, j)], 0.0) - kj);
        }
        let coeff = -2.0 / (k * k) - 2.0 * kj * kj + 2.0 * (Complex64::new(1.0, 0.0) + kj * kj) / k;
        gain.push((g * coeff.powi(-4)).norm());
    }

    (a, Vector::from_vec(gain))
}

/// Compute gammatone filter banks.
///
/// The returned matrix stores one filter per row and FFT bins in columns. Center frequencies are
/// returned in ERB units for the selected conversion approach.
pub fn gammatone_filter_banks(
    opts: &FilterBankOptions,
    order: i32,
    conversion: ErbConversionApproach,
) -> Result<(Matrix, Vector)> {
    let high_freq = checked_high_freq(opts.low_freq, opts.high_freq, opts.fs)?;
    let nfilts = opts.nfilts;
    let nfft = opts.nfft;
    let maxlen = nfft / 2 + 1;
    let t = 1.0 / opts.fs as f64;
    let n = 4.0;
    let fcs = generate_center_frequencies(opts.low_freq, high_freq, nfilts);
    let erb: Vec<f64> = fcs
        .iter()
        .map(|fc| ((fc / EAR_Q).powi(order) + MIN_BW.powi(order)).powf(1.0 / order as f64))
        .collect();
    let b: Vec<f64> = erb
        .iter()
        .map(|v| 1.019 * 2.0 * std::f64::consts::PI * v)
        .collect();
    let mut fbank = Array2::<f64>::zeros((nfilts, maxlen));

    for j in 0..nfilts {
        let fc = fcs[j];
        let bj = b[j];
        let k = (bj * t).exp();
        let cos = (2.0 * fc * std::f64::consts::PI * t).cos();
        let sin = (2.0 * fc * std::f64::consts::PI * t).sin();
        let smax = (3.0 + 2.0_f64.powf(1.5)).sqrt();
        let smin = (3.0 - 2.0_f64.powf(1.5)).sqrt();
        let a = [
            (cos + smax * sin) / k,
            (cos - smax * sin) / k,
            (cos + smin * sin) / k,
            (cos - smin * sin) / k,
        ];
        let wt = 2.0 * fc * std::f64::consts::PI * t;
        let kj = Complex64::from_polar(1.0, wt);
        let gain_g = a.iter().fold(Complex64::new(1.0, 0.0), |acc, av| {
            acc * (2.0 * t * kj * (Complex64::new(*av, 0.0) - kj))
        });
        let coeff = -2.0 / (k * k) - 2.0 * kj * kj + 2.0 * (Complex64::new(1.0, 0.0) + kj * kj) / k;
        let gain = (gain_g * coeff.powi(-4)).norm();
        let pole = Complex64::from_polar(1.0, wt) / (bj * t).exp();

        for i in 0..maxlen {
            let u = Complex64::from_polar(1.0, 2.0 * std::f64::consts::PI * i as f64 / nfft as f64);
            let numerator = a
                .iter()
                .map(|av| (u - Complex64::new(*av, 0.0)).norm())
                .product::<f64>();
            let denom = ((u - pole).norm() * (u - pole.conj()).norm()).powf(n);
            fbank[(j, i)] = (t.powi(4) / gain) * numerator / denom;
        }
        let max = fbank.row(j).iter().copied().fold(0.0, f64::max);
        if max > 0.0 && max.is_finite() {
            for i in 0..maxlen {
                fbank[(j, i)] /= max;
            }
        }
    }

    let fbank = apply_scaling(fbank, opts.scale)?;
    let freqs = fcs.mapv(|f| hz2erb(f, conversion));
    Ok((fbank, freqs))
}

fn apply_scaling(mut fbank: Matrix, scale: Scale) -> Result<Matrix> {
    let scaling = scale_fbank(scale, fbank.nrows());
    for r in 0..fbank.nrows() {
        if let Some(row) = fbank.row_mut(r).as_slice_mut() {
            crate::simd::scale_assign(row, scaling[(r, 0)]);
        } else {
            for c in 0..fbank.ncols() {
                fbank[(r, c)] *= scaling[(r, 0)];
            }
        }
    }
    Ok(fbank)
}
