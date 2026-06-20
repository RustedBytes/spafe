use crate::*;
use ndarray::{Array2, Axis};
use num_complex::Complex64;
use rustfft::FftPlanner;

/// Exception compatibility helpers from the Python package.
pub mod exceptions {
    use super::*;

    /// Return an error when an optional feature is not available.
    pub fn assert_function_availability(available: bool) -> Result<()> {
        if available {
            Ok(())
        } else {
            Err(SpafeError::InvalidParameter("function is not available"))
        }
    }
}

/// Frequency-scale conversion utilities.
pub mod converters {
    use super::*;

    const A: f64 = (1000.0 * std::f64::consts::LN_10) / (24.7 * 4.37);

    /// Convert a frequency in Hertz to ERB.
    pub fn hz2erb(f: f64, _approach: ErbConversionApproach) -> f64 {
        A * (1.0 + f * 0.00437).log10()
    }

    /// Convert an ERB value to Hertz.
    pub fn erb2hz(fe: f64, _approach: ErbConversionApproach) -> f64 {
        (10.0_f64.powf(fe / A) - 1.0) / 0.00437
    }

    /// Convert a frequency in Hertz to Bark.
    pub fn hz2bark(f: f64, approach: BarkConversionApproach) -> f64 {
        match approach {
            BarkConversionApproach::Wang => 6.0 * (f / 600.0).asinh(),
            BarkConversionApproach::Tjomov => 6.7 * ((f + 20.0) / 600.0).asinh(),
            BarkConversionApproach::Schroeder => 7.0 * (f / 650.0).asinh(),
            BarkConversionApproach::Terhardt => 13.3 * ((f * 0.75) / 1000.0).atan(),
            BarkConversionApproach::Zwicker => 8.7 + 14.2 * (f / 1000.0).log10(),
            BarkConversionApproach::Traunmueller => ((26.28 * f) / (1.0 + 1960.0)) - 0.53,
        }
    }

    /// Convert a Bark value to Hertz.
    pub fn bark2hz(fb: f64, approach: BarkConversionApproach) -> f64 {
        match approach {
            BarkConversionApproach::Wang => 600.0 * (fb / 6.0).sinh(),
            BarkConversionApproach::Tjomov => 600.0 * (fb / 6.7).sinh() - 20.0,
            BarkConversionApproach::Schroeder => 650.0 * (fb / 7.0).sinh(),
            BarkConversionApproach::Terhardt => (1000.0 / 0.75) * (fb / 13.0).tan(),
            BarkConversionApproach::Zwicker => 10.0_f64.powf(((fb - 8.7) / 14.2) + 3.0),
            BarkConversionApproach::Traunmueller => {
                let fi = if fb < 2.0 {
                    (fb - 0.3) / 0.85
                } else if fb > 20.1 {
                    (fb + 4.422) / 1.22
                } else {
                    fb
                };
                1960.0 * ((fi + 0.53) / (26.28 - fi))
            }
        }
    }

    /// Convert a frequency in Hertz to Mel.
    pub fn hz2mel(f: f64, approach: MelConversionApproach) -> f64 {
        match approach {
            MelConversionApproach::Oshaghnessy => 2595.0 * (1.0 + f / 700.0).log10(),
            MelConversionApproach::Lindsay => 2410.0 * (1.0 + f / 625.0).log10(),
        }
    }

    /// Convert a Mel value to Hertz.
    pub fn mel2hz(fm: f64, approach: MelConversionApproach) -> f64 {
        match approach {
            MelConversionApproach::Oshaghnessy => 700.0 * (10.0_f64.powf(fm / 2595.0) - 1.0),
            MelConversionApproach::Lindsay => 625.0 * (10.0_f64.powf(fm / 2410.0) - 1.0),
        }
    }
}

/// Filter-bank scaling and RASTA filtering utilities.
pub mod filters {
    use super::*;

    /// Generate a column vector of filter-bank scaling factors.
    pub fn scale_fbank(scale: Scale, nfilts: usize) -> Matrix {
        let mut out = Array2::<f64>::ones((nfilts, 1));
        match scale {
            Scale::Ascendant => {
                for i in 0..nfilts {
                    out[(i, 0)] = (i + 1) as f64 / nfilts as f64;
                }
            }
            Scale::Descendant => {
                for i in 0..nfilts {
                    out[(i, 0)] = (nfilts - i) as f64 / nfilts as f64;
                }
            }
            Scale::Constant => {}
        }
        out
    }

    /// Apply the RASTA filter used by PLP/RPLP feature extraction.
    pub fn rasta_filter(x: &Matrix) -> Matrix {
        let numer: Vec<f64> = (-2..=2).map(|v| -(v as f64) / 10.0).collect();
        let mut y = Array2::<f64>::zeros(x.raw_dim());
        for r in 0..x.nrows() {
            let mut prev_y = 0.0;
            for c in 0..x.ncols() {
                if c < 4 {
                    y[(r, c)] = 0.0;
                    continue;
                }
                let mut acc = 0.0;
                for (k, b) in numer.iter().enumerate() {
                    let idx = c as isize - k as isize;
                    if idx >= 0 {
                        acc += b * x[(r, idx as usize)];
                    }
                }
                let v = acc + 0.94 * prev_y;
                y[(r, c)] = v;
                prev_y = v;
            }
        }
        y
    }
}

/// Signal preprocessing helpers for framing and windowing.
pub mod preprocessing {
    use super::*;

    /// Replace zero values with `f64::EPSILON` before logarithmic operations.
    pub fn zero_handling(x: &Matrix) -> Matrix {
        x.mapv(super::super::replace_zero)
    }

    /// Apply pre-emphasis to an input signal.
    pub fn pre_emphasis(sig: &[f64], pre_emph_coeff: f64) -> Vec<f64> {
        if sig.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(sig.len());
        out.push(sig[0]);
        for i in 1..sig.len() {
            out.push(sig[i] - pre_emph_coeff * sig[i - 1]);
        }
        out
    }

    /// Split a one-dimensional signal into overlapping frames.
    ///
    /// This is the Rust equivalent of the Python NumPy stride-trick helper.
    pub fn stride_trick(a: &[f64], stride_length: usize, stride_step: usize) -> Result<Matrix> {
        if stride_length == 0 || stride_step == 0 || a.len() < stride_length {
            return Err(SpafeError::SignalTooShort);
        }
        let nrows = ((a.len() - stride_length) / stride_step) + 1;
        let mut frames = Array2::<f64>::zeros((nrows, stride_length));
        for r in 0..nrows {
            let start = r * stride_step;
            for c in 0..stride_length {
                frames[(r, c)] = a[start + c];
            }
        }
        Ok(frames)
    }

    /// Transform a signal into overlapping analysis frames.
    pub fn framing(sig: &[f64], fs: usize, win_len: f64, win_hop: f64) -> Result<Matrix> {
        if win_len < win_hop {
            return Err(SpafeError::WindowLength);
        }
        let frame_length = (win_len * fs as f64) as usize;
        let frame_step = (win_hop * fs as f64) as usize;
        if frame_length == 0 || frame_step == 0 || sig.len() < frame_length {
            return Err(SpafeError::SignalTooShort);
        }
        stride_trick(sig, frame_length, frame_step)
    }

    /// Apply a window function to each frame to reduce spectral leakage.
    pub fn windowing(frames: &Matrix, win_type: WindowType) -> Matrix {
        let frame_len = frames.ncols();
        let window = window_values(frame_len, win_type);
        let mut out = frames.clone();
        for mut row in out.axis_iter_mut(Axis(0)) {
            for (sample, w) in row.iter_mut().zip(window.iter()) {
                *sample *= *w;
            }
        }
        out
    }

    /// Generate window coefficients for a frame length and window type.
    pub fn window_values(frame_len: usize, win_type: WindowType) -> Vec<f64> {
        if frame_len <= 1 {
            return vec![1.0; frame_len];
        }
        let m = (frame_len - 1) as f64;
        (0..frame_len)
            .map(|n| {
                let x = 2.0 * std::f64::consts::PI * n as f64 / m;
                match win_type {
                    WindowType::Hanning => 0.5 - 0.5 * x.cos(),
                    WindowType::Bartlet => 2.0 / m * (m / 2.0 - (n as f64 - m / 2.0).abs()),
                    WindowType::Kaiser => kaiser(n, frame_len, 14.0),
                    WindowType::Blackman => 0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos(),
                    WindowType::Hamming => 0.54 - 0.46 * x.cos(),
                }
            })
            .collect()
    }

    fn kaiser(n: usize, len: usize, beta: f64) -> f64 {
        let alpha = (len as f64 - 1.0) / 2.0;
        let ratio = if alpha == 0.0 {
            0.0
        } else {
            (n as f64 - alpha) / alpha
        };
        bessel_i0(beta * (1.0 - ratio * ratio).max(0.0).sqrt()) / bessel_i0(beta)
    }

    fn bessel_i0(x: f64) -> f64 {
        let mut sum = 1.0;
        let mut y = 1.0;
        for k in 1..30 {
            y *= (x * x / 4.0) / ((k * k) as f64);
            sum += y;
        }
        sum
    }
}

/// Cepstral post-processing utilities.
pub mod cepstral {
    use super::*;

    /// Apply cepstral normalization to a matrix of cepstra.
    pub fn normalize_ceps(x: &Matrix, normalization_type: Normalization) -> Matrix {
        match normalization_type {
            Normalization::MeanVariance => {
                let means = x.mean_axis(Axis(0)).unwrap();
                let std = std_all(x);
                let mut out = x.clone();
                for mut row in out.axis_iter_mut(Axis(0)) {
                    for (v, mean) in row.iter_mut().zip(means.iter()) {
                        *v = (*v - mean) / std;
                    }
                }
                out
            }
            Normalization::MeanSubtraction => {
                let means = x.mean_axis(Axis(0)).unwrap();
                let mut out = x.clone();
                for mut row in out.axis_iter_mut(Axis(0)) {
                    for (v, mean) in row.iter_mut().zip(means.iter()) {
                        *v -= mean;
                    }
                }
                out
            }
            Normalization::Variance => x.mapv(|v| v / std_all(x)),
            Normalization::Mean => {
                let mean = x.mean().unwrap_or(0.0);
                let min = x.iter().copied().fold(f64::INFINITY, f64::min);
                let max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                x.mapv(|v| (v - mean) / (max - min))
            }
        }
    }

    /// Apply cepstral liftering to increase or reshape cepstral coefficient magnitudes.
    pub fn lifter_ceps(ceps: &Matrix, lift: i32) -> Matrix {
        if lift == 0 || lift > 10 {
            return ceps.clone();
        }
        let mut out = ceps.clone();
        if lift > 0 {
            for c in 1..out.ncols() {
                let factor = (c as f64).powi(lift);
                for r in 0..out.nrows() {
                    out[(r, c)] *= factor;
                }
            }
        } else {
            let lift = (-lift) as f64;
            for c in 0..out.ncols() {
                let factor =
                    1.0 + (lift / 2.0) * (std::f64::consts::PI * (c + 1) as f64 / lift).sin();
                for r in 0..out.nrows() {
                    out[(r, c)] *= factor;
                }
            }
        }
        out
    }

    /// Calculate delta coefficients with a `w`-point window.
    pub fn deltas(x: &Matrix, w: usize) -> Matrix {
        let hlen = w / 2;
        let mut out = Array2::<f64>::zeros(x.raw_dim());
        for r in 0..x.nrows() {
            for c in 0..x.ncols() {
                let mut acc = 0.0;
                for k in 0..w {
                    let coeff = hlen as isize - k as isize;
                    let padded_idx = c + 2 * hlen - k;
                    let src = if padded_idx < hlen {
                        0
                    } else if padded_idx - hlen >= x.ncols() {
                        x.ncols() - 1
                    } else {
                        padded_idx - hlen
                    };
                    acc += coeff as f64 * x[(r, src)];
                }
                out[(r, c)] = acc;
            }
        }
        out
    }

    fn std_all(x: &Matrix) -> f64 {
        let mean = x.mean().unwrap_or(0.0);
        let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / x.len() as f64;
        let std = var.sqrt();
        if std == 0.0 { f64::EPSILON } else { std }
    }
}

/// Spectral transform utilities.
pub mod spectral {
    use super::*;

    /// Compute one-sided FFT magnitudes for each frame.
    pub fn fft_magnitude(frames: &Matrix, nfft: usize) -> Matrix {
        let bins = nfft / 2 + 1;
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(nfft);
        let mut out = Array2::<f64>::zeros((frames.nrows(), bins));

        for (r, row) in frames.axis_iter(Axis(0)).enumerate() {
            let mut buffer = vec![Complex64::new(0.0, 0.0); nfft];
            for (dst, src) in buffer.iter_mut().zip(row.iter()).take(nfft) {
                dst.re = *src;
            }
            fft.process(&mut buffer);
            for c in 0..bins {
                out[(r, c)] = buffer[c].norm();
            }
        }
        out
    }

    /// Compute one-sided complex FFT bins for each frame.
    pub fn fft_complex(frames: &Matrix, nfft: usize) -> Vec<Vec<Complex64>> {
        let bins = nfft / 2 + 1;
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(nfft);
        let mut out = Vec::with_capacity(frames.nrows());
        for row in frames.axis_iter(Axis(0)) {
            let mut buffer = vec![Complex64::new(0.0, 0.0); nfft];
            for (dst, src) in buffer.iter_mut().zip(row.iter()).take(nfft) {
                dst.re = *src;
            }
            fft.process(&mut buffer);
            out.push(buffer[..bins].to_vec());
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    /// Compute the constant-Q transform.
    ///
    /// The returned outer dimension follows constant-Q pitch bins, each containing one complex value
    /// per input frame.
    pub fn compute_constant_qtransform(
        frames: &Matrix,
        fs: usize,
        low_freq: f64,
        high_freq: Option<f64>,
        nfft: usize,
        number_of_octaves: usize,
        number_of_bins_per_octave: usize,
        win_type: WindowType,
        spectral_threshold: f64,
        f0: f64,
        q_rate: f64,
    ) -> Vec<Vec<Complex64>> {
        let high_freq = high_freq.unwrap_or(fs as f64 / 2.0);
        let tmp_freqs: Vec<f64> = (0..number_of_octaves)
            .flat_map(|m| {
                (0..number_of_bins_per_octave).map(move |n| {
                    f0 * 2.0_f64.powf(
                        (m * number_of_bins_per_octave + n) as f64
                            / number_of_bins_per_octave as f64,
                    )
                })
            })
            .filter(|f| low_freq <= *f && *f <= high_freq)
            .collect();
        let q = q_rate / (2.0_f64.powf(1.0 / number_of_bins_per_octave as f64) - 1.0);
        let mut cqt_freqs = Vec::new();
        let mut win_lens = Vec::new();
        for f in tmp_freqs {
            let nk = (q * fs as f64 / f).ceil() as usize;
            if nk <= nfft {
                cqt_freqs.push(f);
                win_lens.push(nk);
            }
        }

        let fft_frames = fft_complex(frames, nfft);
        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(nfft);
        let mut kernels: Vec<Vec<Complex64>> = Vec::with_capacity(cqt_freqs.len());
        for (fk, nk) in cqt_freqs.iter().zip(win_lens.iter()) {
            let mut a = vec![Complex64::new(0.0, 0.0); nfft];
            let start = (nfft - *nk) / 2;
            let window = super::preprocessing::window_values(*nk, win_type);
            for n in 0..*nk {
                let phase = 2.0 * std::f64::consts::PI * (fk / fs as f64) * n as f64;
                a[start + n] = Complex64::from_polar(window[n] / *nk as f64, phase);
            }
            fft.process(&mut a);
            for z in &mut a {
                if z.norm() <= spectral_threshold {
                    *z = Complex64::new(0.0, 0.0);
                } else {
                    *z = z.conj() / nfft as f64;
                }
            }
            kernels.push(a);
        }

        let mut spec = vec![vec![Complex64::new(0.0, 0.0); frames.nrows()]; kernels.len()];
        for (frame_idx, frame_fft) in fft_frames.iter().enumerate() {
            for (pitch_idx, kernel) in kernels.iter().enumerate() {
                let sum = frame_fft
                    .iter()
                    .zip(kernel.iter())
                    .map(|(x, h)| *x * *h)
                    .fold(Complex64::new(0.0, 0.0), |a, b| a + b);
                spec[pitch_idx][frame_idx] = sum;
            }
        }
        spec
    }
}

/// Visualization helpers that render deterministic SVG strings.
pub mod vis {
    use super::*;
    use crate::utils::converters::{hz2bark, hz2erb, hz2mel};

    /// Return frequency tick labels converted for a filter-bank scale.
    pub fn tick_function(values: &[f64], fb_type: &str) -> Vec<String> {
        values
            .iter()
            .map(|value| match fb_type {
                "mel" => hz2mel(*value, MelConversionApproach::Oshaghnessy),
                "bark" => hz2bark(*value, BarkConversionApproach::Wang),
                "gamma" => hz2erb(*value, ErbConversionApproach::Glasberg),
                _ => *value,
            })
            .map(|value| format!("{value:.1}"))
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    /// Render filter banks as an SVG line plot.
    ///
    /// Each row of `fbanks` is drawn as one filter curve. The returned string is complete SVG markup.
    pub fn show_fbanks(
        fbanks: &Matrix,
        center_freqs: &[f64],
        ref_freqs: &[f64],
        title: &str,
        ylabel: &str,
        x1label: &str,
        x2label: &str,
        fb_type: &str,
        show_center_freqs: bool,
    ) -> Result<String> {
        let mut body = String::new();
        let width = 900.0;
        let height = 360.0;
        let plot = PlotArea::new(width, height);
        let x_min = ref_freqs.first().copied().unwrap_or(0.0);
        let x_max = ref_freqs.last().copied().unwrap_or(1.0);
        let y_max = fbanks.iter().copied().fold(0.0, f64::max).max(1.0);

        for row in fbanks.rows() {
            let points = row
                .iter()
                .enumerate()
                .map(|(idx, value)| {
                    let x = ref_freqs.get(idx).copied().unwrap_or(idx as f64);
                    format!(
                        "{:.3},{:.3}",
                        plot.x(x, x_min, x_max),
                        plot.y(*value, 0.0, y_max)
                    )
                })
                .collect::<Vec<_>>()
                .join(" ");
            body.push_str(&format!(
                r##"<polyline fill="none" stroke="#2563eb" stroke-width="1" points="{points}"/>"##
            ));
        }

        if show_center_freqs {
            for freq in center_freqs {
                let x = plot.x(*freq, x_min, x_max);
                body.push_str(&format!(
                    r##"<line x1="{x:.3}" y1="{:.3}" x2="{x:.3}" y2="{:.3}" stroke="#888" stroke-dasharray="3 3"/>"##,
                    plot.top,
                    plot.bottom
                ));
            }
        }

        let ticks = tick_function(ref_freqs, fb_type).join(", ");
        Ok(svg_document(
            width,
            height,
            title,
            &[
                ylabel,
                x1label,
                x2label,
                &format!("{fb_type} ticks: {ticks}"),
            ],
            &body,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    /// Render a spectrogram matrix as an SVG heatmap.
    pub fn show_spectrogram(
        spectrogram: &Matrix,
        _fs: usize,
        xmin: f64,
        xmax: f64,
        ymin: f64,
        ymax: f64,
        dbf: f64,
        xlabel: &str,
        ylabel: &str,
        title: &str,
        colorbar: bool,
    ) -> Result<String> {
        let magnitude = spectrogram.mapv(f64::abs);
        let ref_value = magnitude.iter().copied().fold(0.0, f64::max).max(1e-10);
        let max_db = 0.0;
        let min_db = max_db - dbf;
        let db = magnitude.mapv(|value| {
            let raw = 10.0 * (value.max(1e-10) / ref_value).log10();
            raw.max(min_db)
        });
        let body = heatmap_svg(&db, min_db, max_db, "#111827", "#f59e0b");
        Ok(svg_document(
            900.0,
            360.0,
            title,
            &[
                xlabel,
                ylabel,
                &format!("extent=({xmin:.3}, {xmax:.3}, {ymin:.3}, {ymax:.3})"),
                if colorbar {
                    "colorbar=true"
                } else {
                    "colorbar=false"
                },
            ],
            &body,
        ))
    }

    /// Render a feature matrix as an SVG heatmap.
    ///
    /// Rows correspond to frames and columns correspond to feature coefficients.
    pub fn show_features(
        features: &Matrix,
        title: &str,
        ylabel: &str,
        xlabel: &str,
    ) -> Result<String> {
        let min = features.iter().copied().fold(f64::INFINITY, f64::min);
        let max = features.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let body = heatmap_svg(&features.t().to_owned(), min, max, "#0f172a", "#38bdf8");
        Ok(svg_document(900.0, 360.0, title, &[ylabel, xlabel], &body))
    }

    struct PlotArea {
        left: f64,
        right: f64,
        top: f64,
        bottom: f64,
    }

    impl PlotArea {
        fn new(width: f64, height: f64) -> Self {
            Self {
                left: 56.0,
                right: width - 24.0,
                top: 42.0,
                bottom: height - 48.0,
            }
        }

        fn x(&self, value: f64, min: f64, max: f64) -> f64 {
            let denom = (max - min).abs().max(f64::EPSILON);
            self.left + ((value - min) / denom).clamp(0.0, 1.0) * (self.right - self.left)
        }

        fn y(&self, value: f64, min: f64, max: f64) -> f64 {
            let denom = (max - min).abs().max(f64::EPSILON);
            self.bottom - ((value - min) / denom).clamp(0.0, 1.0) * (self.bottom - self.top)
        }
    }

    fn heatmap_svg(values: &Matrix, min: f64, max: f64, low: &str, high: &str) -> String {
        let width = 900.0;
        let height = 360.0;
        let plot = PlotArea::new(width, height);
        let cell_w = (plot.right - plot.left) / values.ncols().max(1) as f64;
        let cell_h = (plot.bottom - plot.top) / values.nrows().max(1) as f64;
        let mut body = String::new();
        for row in 0..values.nrows() {
            for col in 0..values.ncols() {
                let color = lerp_color(values[(row, col)], min, max, low, high);
                let x = plot.left + col as f64 * cell_w;
                let y = plot.bottom - (row + 1) as f64 * cell_h;
                body.push_str(&format!(
                    r#"<rect x="{x:.3}" y="{y:.3}" width="{cell_w:.3}" height="{cell_h:.3}" fill="{color}"/>"#
                ));
            }
        }
        body
    }

    fn svg_document(width: f64, height: f64, title: &str, labels: &[&str], body: &str) -> String {
        let escaped_title = escape_xml(title);
        let labels = labels
            .iter()
            .enumerate()
            .map(|(idx, label)| {
                format!(
                    r##"<text x="56" y="{}" font-size="11" fill="#555">{}</text>"##,
                    height - 30.0 + idx as f64 * 13.0,
                    escape_xml(label)
                )
            })
            .collect::<String>();
        format!(
            r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0} {height:.0}" width="{width:.0}" height="{height:.0}"><rect width="100%" height="100%" fill="#fff"/><text x="56" y="24" font-size="18" font-family="sans-serif" fill="#111">{escaped_title}</text>{body}{labels}</svg>"##
        )
    }

    fn lerp_color(value: f64, min: f64, max: f64, low: &str, high: &str) -> String {
        let t = ((value - min) / (max - min).abs().max(f64::EPSILON)).clamp(0.0, 1.0);
        let low = parse_hex(low);
        let high = parse_hex(high);
        format!(
            "#{:02x}{:02x}{:02x}",
            (low.0 as f64 + (high.0 as f64 - low.0 as f64) * t) as u8,
            (low.1 as f64 + (high.1 as f64 - low.1 as f64) * t) as u8,
            (low.2 as f64 + (high.2 as f64 - low.2 as f64) * t) as u8
        )
    }

    fn parse_hex(value: &str) -> (u8, u8, u8) {
        let value = value.trim_start_matches('#');
        let r = u8::from_str_radix(&value[0..2], 16).unwrap_or(0);
        let g = u8::from_str_radix(&value[2..4], 16).unwrap_or(0);
        let b = u8::from_str_radix(&value[4..6], 16).unwrap_or(0);
        (r, g, b)
    }

    fn escape_xml(value: &str) -> String {
        value
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
    }
}
