use crate::*;
use ndarray::Axis;

/// Compute the YIN difference function for a signal frame.
pub fn compute_difference(x: &[f64], tau_max: usize) -> Vector {
    let w = x.len();
    let mut out = Vec::with_capacity(tau_max.min(w));
    for tau in 0..tau_max.min(w) {
        let mut sum = 0.0;
        for j in 0..w - tau {
            let d = x[j] - x[j + tau];
            sum += d * d;
        }
        out.push(sum);
    }
    Vector::from_vec(out)
}

/// Apply the cumulative mean normalized difference function (CMNDF).
pub fn compute_cmnd(d_t: &[f64], tau: usize) -> Vector {
    let mut out = Vec::with_capacity(tau.min(d_t.len()));
    out.push(1.0);
    let mut cumsum = 0.0;
    for (i, value) in d_t.iter().enumerate().take(tau.min(d_t.len())).skip(1) {
        cumsum += value;
        out.push(value * i as f64 / cumsum.max(f64::EPSILON));
    }
    Vector::from_vec(out)
}

/// Return the fundamental period of a frame from a CMND function.
///
/// Returns `0.0` when no candidate falls below `harmonic_threshold`.
pub fn get_pitch(cmdf: &[f64], tau_min: usize, tau_max: usize, harmonic_threshold: f64) -> f64 {
    let mut tau = tau_min;
    let tau_max = tau_max.min(cmdf.len());
    while tau < tau_max {
        if cmdf[tau] < harmonic_threshold {
            while tau + 1 < tau_max && cmdf[tau + 1] < cmdf[tau] {
                tau += 1;
            }
            return tau as f64;
        }
        tau += 1;
    }
    0.0
}

/// Compute fundamental frequency and harmonic-rate tracks with the YIN algorithm.
///
/// Returns `(pitches, harmonic_rates, argmins, times)`.
pub fn compute_yin(
    sig: &[f64],
    fs: usize,
    win_len: f64,
    win_hop: f64,
    low_freq: f64,
    high_freq: f64,
    harmonic_threshold: f64,
) -> Result<(Vector, Vector, Vector, Vector)> {
    let frames = crate::utils::preprocessing::framing(sig, fs, win_len, win_hop)?;
    let tau_min = (fs as f64 / high_freq) as usize;
    let tau_max = (fs as f64 / low_freq) as usize;
    let mut pitches = Vec::new();
    let mut harmonic_rates = Vec::new();
    let mut argmins = Vec::new();
    let mut times = Vec::new();
    for (i, frame) in frames.axis_iter(Axis(0)).enumerate() {
        let data = frame.to_vec();
        let diff = compute_difference(&data, tau_max);
        let cmnd = compute_cmnd(diff.as_slice().unwrap(), tau_max);
        let pitch_tau = get_pitch(
            cmnd.as_slice().unwrap(),
            tau_min,
            tau_max,
            harmonic_threshold,
        );
        if pitch_tau != 0.0 {
            pitches.push(fs as f64 / pitch_tau);
            harmonic_rates.push(cmnd[pitch_tau as usize]);
        } else {
            pitches.push(0.0);
            harmonic_rates.push(1.0);
        }
        let (argmin, minval) = cmnd
            .iter()
            .enumerate()
            .skip(tau_min)
            .take(tau_max.saturating_sub(tau_min))
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, value)| (idx as f64, *value))
            .unwrap_or((0.0, 1.0));
        argmins.push(if minval.is_finite() { argmin } else { 0.0 });
        times.push(i as f64 * win_hop);
    }
    Ok((
        Vector::from_vec(pitches),
        Vector::from_vec(harmonic_rates),
        Vector::from_vec(argmins),
        Vector::from_vec(times),
    ))
}

/// Return dominant frequencies for each frame of an audio signal.
pub fn get_dominant_frequencies(
    sig: &[f64],
    fs: usize,
    nfft: usize,
    win_len: f64,
    win_hop: f64,
    win_type: WindowType,
    only_positive: bool,
) -> Result<Vector> {
    let frames = crate::utils::preprocessing::framing(sig, fs, win_len, win_hop)?;
    let windows = crate::utils::preprocessing::windowing(&frames, win_type);
    let mags = crate::utils::spectral::fft_magnitude(&windows, nfft);
    let mut freqs = Vec::new();
    for row in mags.axis_iter(Axis(0)) {
        let (idx, _) = row
            .iter()
            .enumerate()
            .map(|(i, v)| (i, (v / nfft as f64).powi(2) / nfft as f64))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        let freq = idx as f64 * fs as f64 / row.len() as f64;
        let rounded = (freq * 1000.0).round() / 1000.0;
        if !only_positive || rounded >= 0.0 {
            freqs.push(rounded);
        }
    }
    Ok(Vector::from_vec(freqs))
}
