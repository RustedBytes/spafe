use crate::fbanks::*;
use crate::*;
use ndarray::{Array2, Axis};
use num_complex::Complex64;
use rustfft::FftPlanner;

/// Compute the Mel-scale spectrogram for an audio signal.
pub fn mel_spectrogram(sig: &[f64], opts: &FeatureOptions) -> Result<SpectrogramOutput> {
    let fopts = FilterBankOptions::from(opts);
    let (fbanks, _) = mel_filter_banks(&fopts, MelConversionApproach::Oshaghnessy)?;
    spectrogram_with_fbanks(sig, opts, &fbanks)
}

/// Compute Mel-frequency cepstral coefficients (MFCCs) from an audio signal.
///
/// Options control pre-emphasis, framing, filter-bank size, DCT type, liftering, and normalization.
pub fn mfcc(sig: &[f64], opts: &FeatureOptions) -> Result<Matrix> {
    if opts.nfilts < opts.num_ceps {
        return Err(SpafeError::FilterCount);
    }
    let spec = mel_spectrogram(sig, opts)?;
    Ok(cepstral_from_spectrogram(spec, opts, f64::ln))
}

/// Compute inverse Mel-frequency cepstral coefficients (IMFCCs) from an audio signal.
pub fn imfcc(sig: &[f64], opts: &FeatureOptions) -> Result<Matrix> {
    if opts.nfilts < opts.num_ceps {
        return Err(SpafeError::FilterCount);
    }
    let fopts = FilterBankOptions::from(opts);
    let (fbanks, _) = inverse_mel_filter_banks(&fopts, MelConversionApproach::Oshaghnessy)?;
    let spec = spectrogram_with_fbanks(sig, opts, &fbanks)?;
    Ok(cepstral_from_spectrogram(spec, opts, f64::ln))
}

/// Compute a linear-frequency spectrogram from an audio signal.
pub fn linear_spectrogram(sig: &[f64], opts: &FeatureOptions) -> Result<SpectrogramOutput> {
    let fopts = FilterBankOptions::from(opts);
    let (fbanks, _) = linear_filter_banks(&fopts)?;
    spectrogram_with_fbanks(sig, opts, &fbanks)
}

/// Compute linear-frequency cepstral coefficients (LFCCs) from an audio signal.
pub fn lfcc(sig: &[f64], opts: &FeatureOptions) -> Result<Matrix> {
    if opts.nfilts < opts.num_ceps {
        return Err(SpafeError::FilterCount);
    }
    let spec = linear_spectrogram(sig, opts)?;
    Ok(cepstral_from_spectrogram(spec, opts, f64::ln))
}

/// Compute the Bark-scale spectrogram for an audio signal.
pub fn bark_spectrogram(sig: &[f64], opts: &FeatureOptions) -> Result<SpectrogramOutput> {
    let fopts = FilterBankOptions::from(opts);
    let (fbanks, _) = bark_filter_banks(&fopts, BarkConversionApproach::Wang)?;
    spectrogram_with_fbanks(sig, opts, &fbanks)
}

/// Apply the intensity power law used by Bark-frequency cepstral features.
pub fn intensity_power_law(w: &Matrix) -> Matrix {
    w.mapv(|v| {
        let f = |x: f64, c: f64, p: i32| x.powi(2) + c * 10.0_f64.powi(p);
        let e =
            (f(v, 56.8, 6) * v.powi(4)) / (f(v, 6.3, 6) * f(v, 0.38, 9) * f(v.powi(3), 9.58, 26));
        e.powf(1.0 / 3.0)
    })
}

/// Compute Bark-frequency cepstral coefficients (BFCCs) from an audio signal.
pub fn bfcc(sig: &[f64], opts: &FeatureOptions) -> Result<Matrix> {
    if opts.nfilts < opts.num_ceps {
        return Err(SpafeError::FilterCount);
    }
    let mut spec = bark_spectrogram(sig, opts)?;
    spec.features = intensity_power_law(&spec.features);
    Ok(cepstral_from_spectrogram(spec, opts, f64::ln))
}

/// Compute the gammatone/ERB spectrogram, also known as a cochleagram.
pub fn erb_spectrogram(sig: &[f64], opts: &FeatureOptions) -> Result<SpectrogramOutput> {
    let fopts = FilterBankOptions::from(opts);
    let (fbanks, _) = gammatone_filter_banks(&fopts, 4, ErbConversionApproach::Glasberg)?;
    spectrogram_with_fbanks(sig, opts, &fbanks)
}

/// Compute gammatone-frequency cepstral coefficients (GFCCs) from an audio signal.
pub fn gfcc(sig: &[f64], opts: &FeatureOptions) -> Result<Matrix> {
    if opts.nfilts < opts.num_ceps {
        return Err(SpafeError::FilterCount);
    }
    let spec = erb_spectrogram(sig, opts)?;
    Ok(cepstral_from_spectrogram(spec, opts, |v| v.powf(1.0 / 3.0)))
}

/// Compute magnitude-based spectral root cepstral coefficients (MSRCCs).
pub fn msrcc(sig: &[f64], opts: &FeatureOptions, gamma: f64) -> Result<Matrix> {
    if opts.nfilts < opts.num_ceps {
        return Err(SpafeError::FilterCount);
    }
    let spec = mel_spectrogram(sig, opts)?;
    Ok(cepstral_from_spectrogram(spec, opts, |v| v.powf(gamma)))
}

/// Compute normalized gammachirp cepstral coefficients (NGCCs).
pub fn ngcc(sig: &[f64], opts: &FeatureOptions) -> Result<Matrix> {
    if opts.nfilts < opts.num_ceps {
        return Err(SpafeError::FilterCount);
    }
    let spec = erb_spectrogram(sig, opts)?;
    Ok(cepstral_from_spectrogram(spec, opts, f64::ln))
}

/// Compute phase-based spectral root cepstral coefficients (PSRCCs).
pub fn psrcc(sig: &[f64], opts: &FeatureOptions, gamma: f64) -> Result<Matrix> {
    if opts.nfilts < opts.num_ceps {
        return Err(SpafeError::FilterCount);
    }
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
    let fft = crate::utils::spectral::fft_complex(&windows, opts.nfft);
    let mut phases = Array2::<f64>::zeros((fft.len(), opts.nfft / 2 + 1));
    for (r, row) in fft.iter().enumerate() {
        for (c, z) in row.iter().enumerate() {
            let deg = z.arg().to_degrees();
            phases[(r, c)] = if deg < 0.0 { 360.0 + deg } else { deg };
        }
    }
    let fopts = FilterBankOptions::from(opts);
    let (fbanks, _) = mel_filter_banks(&fopts, MelConversionApproach::Oshaghnessy)?;
    let mut features = phases.dot(&fbanks.t()).mapv(|v| v.powf(gamma));
    let max = features
        .iter()
        .filter(|v| v.is_finite())
        .copied()
        .fold(0.0, f64::max);
    features.mapv_inplace(|v| {
        if v.is_nan() {
            0.0
        } else if v.is_infinite() {
            max
        } else {
            v
        }
    });
    let spec = SpectrogramOutput {
        features,
        fft_magnitude: phases,
    };
    Ok(cepstral_from_spectrogram(spec, opts, |v| v))
}

/// Compute medium-time power for PNCC processing.
pub fn medium_time_power_calculation(p: &Matrix, m_window: usize) -> Matrix {
    let mut out = Array2::<f64>::zeros(p.raw_dim());
    for m in 0..p.nrows() {
        let start = m.saturating_sub(m_window);
        let end = (m + m_window + 1).min(p.nrows());
        for l in 0..p.ncols() {
            out[(m, l)] = p.slice(ndarray::s![start..end, l]).sum() / (2 * m_window + 1) as f64;
        }
    }
    out
}

/// Apply asymmetric low-pass filtering for PNCC noise tracking.
pub fn asymmetric_lowpass_filtering(q_in: &Matrix, lm_a: f64, lm_b: f64) -> Matrix {
    let mut out = Array2::<f64>::zeros(q_in.raw_dim());
    if q_in.nrows() == 0 {
        return out;
    }
    for l in 0..q_in.ncols() {
        out[(0, l)] = 0.9 * q_in[(0, l)];
    }
    for m in 0..q_in.nrows() {
        for l in 0..q_in.ncols() {
            let prev = if m == 0 { 0.0 } else { out[(m - 1, l)] };
            let q1 = lm_a * prev + (1.0 - lm_a) * q_in[(m, l)];
            let q2 = lm_b * prev + (1.0 - lm_b) * q_in[(m, l)];
            out[(m, l)] = if q_in[(m, l)] >= prev { q1 } else { q2 };
        }
    }
    out
}

/// Apply temporal masking to a rectified PNCC signal.
pub fn temporal_masking(q0: &Matrix, lam_t: f64, myu_t: f64) -> Matrix {
    let mut q_tm = Array2::<f64>::zeros(q0.raw_dim());
    let mut peak = Array2::<f64>::zeros(q0.raw_dim());
    if q0.nrows() == 0 {
        return q_tm;
    }
    for l in 0..q0.ncols() {
        q_tm[(0, l)] = q0[(0, l)];
        peak[(0, l)] = q0[(0, l)];
    }
    for m in 1..q0.nrows() {
        for l in 0..q0.ncols() {
            peak[(m, l)] = (lam_t * peak[(m - 1, l)]).max(q0[(m, l)]);
            q_tm[(m, l)] = if q0[(m, l)] >= lam_t * peak[(m - 1, l)] {
                q0[(m, l)]
            } else {
                myu_t * peak[(m - 1, l)]
            };
        }
    }
    q_tm
}

/// Apply PNCC spectral weight smoothing.
pub fn weight_smoothing(r: &Matrix, q: &Matrix, nfilts: usize, n: usize) -> Matrix {
    let mut out = Array2::<f64>::zeros(r.raw_dim());
    for m in 0..r.nrows() {
        for l in 0..r.ncols() {
            let l1 = l.saturating_sub(n).max(1);
            let l2 = (l + n).min(nfilts).min(r.ncols());
            let width = (l2.saturating_sub(l1) + 1).max(1) as f64;
            let mut sum = 0.0;
            for lp in l1..l2 {
                sum += r[(m, lp)] / q[(m, lp)].max(f64::EPSILON);
            }
            out[(m, l)] = sum / width;
        }
    }
    out
}

/// Apply PNCC mean power normalization.
pub fn mean_power_normalization(t: &Matrix, lam_myu: f64, nfilts: usize, k: f64) -> Matrix {
    let mut myu = vec![0.0001; t.nrows()];
    for m in 1..t.nrows() {
        let row_sum = t.row(m).sum();
        myu[m] = lam_myu * myu[m - 1] + ((1.0 - lam_myu) / nfilts as f64) * row_sum;
    }
    let mut out = t.clone();
    for m in 0..out.nrows() {
        for l in 0..out.ncols() {
            out[(m, l)] = k * out[(m, l)] / myu[m].max(f64::EPSILON);
        }
    }
    out
}

/// Apply asymmetric noise suppression with temporal masking for PNCCs.
pub fn asymmetric_noise_suppression_with_temporal_masking(
    q_tilde: &Matrix,
    threshold: f64,
) -> Matrix {
    let q_le = asymmetric_lowpass_filtering(q_tilde, 0.999, 0.5);
    let mut q0 = q_tilde - &q_le;
    q0.mapv_inplace(|v| if v < threshold { 0.0 } else { v });
    let q_f = asymmetric_lowpass_filtering(&q0, 0.999, 0.5);
    let q_tm = temporal_masking(&q0, 0.85, 0.2);
    let mut r = Array2::<f64>::zeros(q_tilde.raw_dim());
    for m in 0..q_tilde.nrows() {
        for l in 0..q_tilde.ncols() {
            r[(m, l)] = if q_tilde[(m, l)] >= 2.0 * q_le[(m, l)] {
                q_tm[(m, l)]
            } else {
                q_f[(m, l)]
            };
        }
    }
    r
}

/// Compute power-normalized cepstral coefficients (PNCCs) from an audio signal.
pub fn pncc(sig: &[f64], opts: &FeatureOptions, power: f64) -> Result<Matrix> {
    if opts.nfilts < opts.num_ceps {
        return Err(SpafeError::FilterCount);
    }
    let fopts = FilterBankOptions::from(opts);
    let (fbanks, _) = gammatone_filter_banks(&fopts, 4, ErbConversionApproach::Glasberg)?;
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
    let fft = crate::utils::spectral::fft_magnitude(&windows, opts.nfft);
    let p = fft
        .mapv(|v| v.powf(power) / opts.nfft as f64)
        .dot(&fbanks.t());
    let q = medium_time_power_calculation(&p, 2);
    let r = asymmetric_noise_suppression_with_temporal_masking(&q, 0.0);
    let s = weight_smoothing(&r, &q, opts.nfilts, 4);
    let t = p * s;
    let u = mean_power_normalization(&t, 0.999, opts.nfilts, 1.0);
    let v = u.mapv(|x| x.max(f64::EPSILON).powf(1.0 / 15.0));
    let ceps = dct_rows(&v, opts.dct_type, opts.num_ceps);
    Ok(apply_post_processing(ceps, opts))
}

/// Compute linear prediction coefficients (LPCs) for each frame of an audio signal.
///
/// Returns LPC coefficient rows and one prediction-error value per frame.
pub fn lpc(
    sig: &[f64],
    fs: usize,
    order: usize,
    pre_emph: bool,
    window: SlidingWindow,
) -> Result<(Matrix, Vector)> {
    let lp_order = order.saturating_sub(1);
    let signal = if pre_emph {
        crate::utils::preprocessing::pre_emphasis(sig, 0.97)
    } else {
        sig.to_vec()
    };
    let frames = crate::utils::preprocessing::framing(&signal, fs, window.win_len, window.win_hop)?;
    let mut a_mat = Array2::<f64>::zeros((frames.nrows(), lp_order + 1));
    let mut e_vec = Vec::with_capacity(frames.nrows());
    for (i, frame) in frames.axis_iter(Axis(0)).enumerate() {
        let (a, e) = lpc_helper(frame.as_slice().unwrap(), lp_order);
        for (j, coeff) in a.iter().enumerate() {
            a_mat[(i, j)] = *coeff;
        }
        e_vec.push(e);
    }
    Ok((a_mat, Vector::from_vec(e_vec)))
}

/// Convert linear prediction coefficients (LPCs) to linear prediction cepstral coefficients.
pub fn lpc2lpcc(a: &[f64], e: f64, nceps: usize) -> Vec<f64> {
    let p = a.len();
    let mut c = vec![1.0; nceps];
    if nceps == 0 {
        return c;
    }
    c[0] = replace_zero(e).ln();
    for m in 1..p.min(nceps) {
        let mut sum = 0.0;
        for k in 1..m {
            sum += (k as f64 / m as f64) * c[m] * a[m - k];
        }
        c[m] = a[m] + sum;
    }
    if nceps > p {
        for m in p + 1..nceps {
            let mut sum = 0.0;
            for k in m - p..m {
                sum += (k as f64 / m as f64) * c[m] * a[m - k];
            }
            c[m] = sum;
        }
    }
    c
}

/// Compute linear predictive cepstral coefficients (LPCCs) from an audio signal.
pub fn lpcc(sig: &[f64], opts: &FeatureOptions) -> Result<Matrix> {
    let (a, e) = lpc(sig, opts.fs, opts.num_ceps, opts.pre_emph, opts.window)?;
    let mut out = Array2::<f64>::zeros((a.nrows(), opts.num_ceps));
    for r in 0..a.nrows() {
        let coeffs = lpc2lpcc(a.row(r).as_slice().unwrap(), e[r], opts.num_ceps);
        for c in 0..opts.num_ceps {
            out[(r, c)] = coeffs[c];
        }
    }
    Ok(apply_post_processing(out, opts))
}

/// Compute perceptual linear prediction (PLP) coefficients.
///
/// Set `do_rasta` to apply RASTA filtering before the perceptual prediction stage.
pub fn plp(sig: &[f64], opts: &FeatureOptions, do_rasta: bool) -> Result<Matrix> {
    if opts.nfilts < opts.num_ceps {
        return Err(SpafeError::FilterCount);
    }
    let mut spec = bark_spectrogram(sig, opts)?.features;
    if do_rasta {
        spec.mapv_inplace(|v| replace_zero(v).ln());
        spec = crate::utils::filters::rasta_filter(&spec);
        spec.mapv_inplace(f64::exp);
    }
    let equal_loudness = |w: f64| {
        ((w.powi(2) + 56.8 * 10.0_f64.powi(6)) * w.powi(4))
            / ((w.powi(2) + 6.3 * 10.0_f64.powi(6))
                * (w.powi(2) + 0.38 * 10.0_f64.powi(9))
                * (w.powi(6) + 9.58 * 10.0_f64.powi(26)))
    };
    let l = spec.mapv(|v| equal_loudness(v).abs().powf(1.0 / 3.0));
    let ifft = ifft_abs_rows(&l, opts.nfft);
    let mut lpccs = Array2::<f64>::zeros((l.nrows(), opts.num_ceps));
    for i in 0..l.nrows() {
        let (a, e) = lpc_helper(ifft.row(i).as_slice().unwrap(), opts.num_ceps - 1);
        let coeffs = lpc2lpcc(&a, e, opts.num_ceps);
        for c in 0..opts.num_ceps {
            lpccs[(i, c)] = coeffs[c];
        }
    }
    Ok(apply_post_processing(lpccs, opts))
}

/// Compute RASTA perceptual linear prediction (RPLP) coefficients.
pub fn rplp(sig: &[f64], opts: &FeatureOptions) -> Result<Matrix> {
    plp(sig, opts, true)
}

/// Compute a constant-Q spectrogram from an audio signal.
///
/// The output contains one row per frame and one column per retained constant-Q bin.
pub fn cqt_spectrogram(
    sig: &[f64],
    opts: &FeatureOptions,
    number_of_octaves: usize,
    number_of_bins_per_octave: usize,
    spectral_threshold: f64,
    f0: f64,
    q_rate: f64,
) -> Result<Matrix> {
    let high_freq = checked_high_freq(opts.low_freq, opts.high_freq, opts.fs)?;
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
    let tmp_freqs: Vec<f64> = (0..number_of_octaves)
        .flat_map(|m| {
            (0..number_of_bins_per_octave).map(move |n| {
                f0 * 2.0_f64.powf(
                    (m * number_of_bins_per_octave + n) as f64 / number_of_bins_per_octave as f64,
                )
            })
        })
        .filter(|f| opts.low_freq <= *f && *f <= high_freq)
        .collect();
    let q = q_rate / (2.0_f64.powf(1.0 / number_of_bins_per_octave as f64) - 1.0);
    let mut cqt_freqs = Vec::new();
    let mut win_lens = Vec::new();
    for f in tmp_freqs {
        let nk = (q * opts.fs as f64 / f).ceil() as usize;
        if nk <= opts.nfft {
            cqt_freqs.push(f);
            win_lens.push(nk);
        }
    }

    let fft_frames = crate::utils::spectral::fft_complex(&windows, opts.nfft);
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(opts.nfft);
    let mut kernels: Vec<Vec<Complex64>> = Vec::with_capacity(cqt_freqs.len());
    for (fk, nk) in cqt_freqs.iter().zip(win_lens.iter()) {
        let mut a = vec![Complex64::new(0.0, 0.0); opts.nfft];
        let start = (opts.nfft - *nk) / 2;
        let window = crate::utils::preprocessing::window_values(*nk, opts.window.win_type);
        for n in 0..*nk {
            let phase = 2.0 * std::f64::consts::PI * (fk / opts.fs as f64) * n as f64;
            a[start + n] = Complex64::from_polar(window[n] / *nk as f64, phase);
        }
        fft.process(&mut a);
        for z in &mut a {
            if z.norm() <= spectral_threshold {
                *z = Complex64::new(0.0, 0.0);
            } else {
                *z = z.conj() / opts.nfft as f64;
            }
        }
        kernels.push(a);
    }

    let mut spec = Array2::<f64>::zeros((fft_frames.len(), kernels.len()));
    for (r, frame_fft) in fft_frames.iter().enumerate() {
        for (k, kernel) in kernels.iter().enumerate() {
            let sum = frame_fft
                .iter()
                .zip(kernel.iter())
                .map(|(x, h)| *x * *h)
                .fold(Complex64::new(0.0, 0.0), |a, b| a + b);
            spec[(r, k)] = sum.norm();
        }
    }
    Ok(spec)
}

/// Compute constant-Q cepstral coefficients (CQCCs) from an audio signal.
pub fn cqcc(sig: &[f64], opts: &FeatureOptions, cqcc_opts: &CqccOptions) -> Result<Matrix> {
    let cqt = cqt_spectrogram(
        sig,
        opts,
        cqcc_opts.number_of_octaves,
        cqcc_opts.number_of_bins_per_octave,
        cqcc_opts.spectral_threshold,
        cqcc_opts.f0,
        cqcc_opts.q_rate,
    )?;
    let log_features = cqt.mapv(|v| replace_zero(v * v).ln());
    let target_rows =
        ((log_features.nrows() as f64) * cqcc_opts.resampling_ratio).max(1.0) as usize;
    let resampled = linear_resample_rows(&log_features, target_rows);
    let ceps = dct_rows(&resampled, opts.dct_type, opts.num_ceps);
    Ok(apply_post_processing(ceps, opts))
}

fn lpc_helper(frame: &[f64], order: usize) -> (Vec<f64>, f64) {
    let p = order + 1;
    let auto_corr = full_autocorr(frame);
    let mut r = vec![0.0; p];
    let nx = p.min(frame.len());
    let center = frame.len().saturating_sub(1);
    r[..nx].copy_from_slice(&auto_corr[center..center + nx]);

    let mut toeplitz = vec![vec![0.0; order]; order];
    for i in 0..order {
        for j in 0..order {
            toeplitz[i][j] = r[i.abs_diff(j)];
        }
    }
    let rhs: Vec<f64> = r[1..].iter().map(|v| -*v).collect();
    let phi = solve_linear(toeplitz, rhs).unwrap_or_else(|| vec![0.0; order]);
    let mut a = Vec::with_capacity(p);
    a.push(1.0);
    a.extend(phi);

    let e = auto_corr[0]
        + auto_corr
            .iter()
            .skip(1)
            .zip(a.iter())
            .map(|(ac, coeff)| ac * coeff)
            .sum::<f64>();
    (a, e.powi(2).sqrt())
}

fn full_autocorr(frame: &[f64]) -> Vec<f64> {
    if frame.is_empty() {
        return Vec::new();
    }
    let n = frame.len();
    let mut out = vec![0.0; 2 * n - 1];
    for (idx, value) in out.iter_mut().enumerate() {
        let lag = idx as isize - (n as isize - 1);
        let mut sum = 0.0;
        for i in 0..n {
            let j = i as isize + lag;
            if (0..n as isize).contains(&j) {
                sum += frame[i] * frame[j as usize];
            }
        }
        *value = sum;
    }
    out
}

fn solve_linear(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for col in 0..n {
        let mut pivot = col;
        for row in col + 1..n {
            if a[row][col].abs() > a[pivot][col].abs() {
                pivot = row;
            }
        }
        if a[pivot][col] == 0.0 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);

        let div = a[col][col];
        for value in a[col].iter_mut().take(n).skip(col) {
            *value /= div;
        }
        b[col] /= div;

        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            let pivot_tail: Vec<f64> = a[col][col..n].to_vec();
            for (value, pivot_value) in a[row].iter_mut().skip(col).zip(pivot_tail.iter()) {
                *value -= factor * pivot_value;
            }
            b[row] -= factor * b[col];
        }
    }
    Some(b)
}

fn ifft_abs_rows(input: &Matrix, nfft: usize) -> Matrix {
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_inverse(nfft);
    let mut out = Array2::<f64>::zeros((input.nrows(), nfft));
    for (r, row) in input.axis_iter(Axis(0)).enumerate() {
        let mut buffer = vec![Complex64::new(0.0, 0.0); nfft];
        for (dst, src) in buffer.iter_mut().zip(row.iter()).take(nfft) {
            dst.re = *src;
        }
        fft.process(&mut buffer);
        for c in 0..nfft {
            out[(r, c)] = (buffer[c] / nfft as f64).norm();
        }
    }
    out
}

fn linear_resample_rows(x: &Matrix, rows: usize) -> Matrix {
    if rows == x.nrows() {
        return x.clone();
    }
    let mut out = Array2::<f64>::zeros((rows, x.ncols()));
    if rows == 1 || x.nrows() == 1 {
        out.row_mut(0).assign(&x.row(0));
        return out;
    }
    let scale = (x.nrows() - 1) as f64 / (rows - 1) as f64;
    for r in 0..rows {
        let pos = r as f64 * scale;
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        let t = pos - lo as f64;
        for c in 0..x.ncols() {
            out[(r, c)] = x[(lo, c)] * (1.0 - t) + x[(hi, c)] * t;
        }
    }
    out
}
