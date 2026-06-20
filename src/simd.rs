#[cfg(feature = "portable-simd")]
mod imp {
    use num_complex::Complex64;
    use std::simd::num::SimdFloat;
    use std::simd::{Simd, StdFloat};

    const LANES: usize = 4;

    #[inline]
    pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
        debug_assert_eq!(a.len(), b.len());
        let len = a.len().min(b.len());
        let mut acc = Simd::<f64, LANES>::splat(0.0);
        let chunks = len / LANES;
        for idx in 0..chunks {
            let start = idx * LANES;
            acc += Simd::from_slice(&a[start..start + LANES])
                * Simd::from_slice(&b[start..start + LANES]);
        }
        acc.reduce_sum()
            + a[chunks * LANES..len]
                .iter()
                .zip(&b[chunks * LANES..len])
                .map(|(x, y)| x * y)
                .sum::<f64>()
    }

    #[inline]
    pub(crate) fn sum_squares_scaled(values: &[f64], scale: f64) -> f64 {
        let mut acc = Simd::<f64, LANES>::splat(0.0);
        let mut chunks = values.chunks_exact(LANES);
        for chunk in &mut chunks {
            let v = Simd::<f64, LANES>::from_slice(chunk);
            acc += v * v;
        }
        (acc.reduce_sum() + chunks.remainder().iter().map(|v| v * v).sum::<f64>()) * scale
    }

    #[inline]
    pub(crate) fn sum_squared_diff(a: &[f64], b: &[f64]) -> f64 {
        debug_assert_eq!(a.len(), b.len());
        let len = a.len().min(b.len());
        let mut acc = Simd::<f64, LANES>::splat(0.0);
        let chunks = len / LANES;
        for idx in 0..chunks {
            let start = idx * LANES;
            let d = Simd::<f64, LANES>::from_slice(&a[start..start + LANES])
                - Simd::<f64, LANES>::from_slice(&b[start..start + LANES]);
            acc += d * d;
        }
        acc.reduce_sum()
            + a[chunks * LANES..len]
                .iter()
                .zip(&b[chunks * LANES..len])
                .map(|(x, y)| {
                    let d = x - y;
                    d * d
                })
                .sum::<f64>()
    }

    #[inline]
    pub(crate) fn sum_squared_offset(values: &[f64], offset: f64) -> f64 {
        let offset = Simd::<f64, LANES>::splat(offset);
        let mut acc = Simd::<f64, LANES>::splat(0.0);
        let mut chunks = values.chunks_exact(LANES);
        for chunk in &mut chunks {
            let d = Simd::<f64, LANES>::from_slice(chunk) - offset;
            acc += d * d;
        }
        acc.reduce_sum()
            + chunks
                .remainder()
                .iter()
                .map(|value| {
                    let d = value - offset[0];
                    d * d
                })
                .sum::<f64>()
    }

    #[inline]
    pub(crate) fn dot_square_weighted(values: &[f64], weights: &[f64], scale: f64) -> f64 {
        debug_assert_eq!(values.len(), weights.len());
        let len = values.len().min(weights.len());
        let mut acc = Simd::<f64, LANES>::splat(0.0);
        let chunks = len / LANES;
        for idx in 0..chunks {
            let start = idx * LANES;
            let v = Simd::<f64, LANES>::from_slice(&values[start..start + LANES]);
            let w = Simd::<f64, LANES>::from_slice(&weights[start..start + LANES]);
            acc += v * v * w;
        }
        (acc.reduce_sum()
            + values[chunks * LANES..len]
                .iter()
                .zip(&weights[chunks * LANES..len])
                .map(|(value, weight)| value * value * weight)
                .sum::<f64>())
            * scale
    }

    #[inline]
    pub(crate) fn mul_assign(values: &mut [f64], weights: &[f64]) {
        debug_assert_eq!(values.len(), weights.len());
        let len = values.len().min(weights.len());
        let chunks = len / LANES;
        for idx in 0..chunks {
            let start = idx * LANES;
            let out = Simd::<f64, LANES>::from_slice(&values[start..start + LANES])
                * Simd::<f64, LANES>::from_slice(&weights[start..start + LANES]);
            values[start..start + LANES].copy_from_slice(&out.to_array());
        }
        for (value, weight) in values[chunks * LANES..len]
            .iter_mut()
            .zip(&weights[chunks * LANES..len])
        {
            *value *= *weight;
        }
    }

    #[inline]
    pub(crate) fn scale_assign(values: &mut [f64], scale: f64) {
        let scale = Simd::<f64, LANES>::splat(scale);
        let mut chunks = values.chunks_exact_mut(LANES);
        for chunk in &mut chunks {
            let out = Simd::<f64, LANES>::from_slice(chunk) * scale;
            chunk.copy_from_slice(&out.to_array());
        }
        for value in chunks.into_remainder() {
            *value *= scale[0];
        }
    }

    #[inline]
    pub(crate) fn pre_emphasis(sig: &[f64], coeff: f64) -> Vec<f64> {
        if sig.is_empty() {
            return Vec::new();
        }
        let mut out = vec![0.0; sig.len()];
        out[0] = sig[0];
        let coeff = Simd::<f64, LANES>::splat(coeff);
        let len = sig.len() - 1;
        let chunks = len / LANES;
        for idx in 0..chunks {
            let start = idx * LANES;
            let cur = Simd::<f64, LANES>::from_slice(&sig[start + 1..start + 1 + LANES]);
            let prev = Simd::<f64, LANES>::from_slice(&sig[start..start + LANES]);
            let values = cur - coeff * prev;
            out[start + 1..start + 1 + LANES].copy_from_slice(&values.to_array());
        }
        for idx in chunks * LANES + 1..sig.len() {
            out[idx] = sig[idx] - coeff[0] * sig[idx - 1];
        }
        out
    }

    #[inline]
    pub(crate) fn complex_mul_real(spectrum: &[Complex64], weights: &[f64]) -> Vec<Complex64> {
        debug_assert_eq!(spectrum.len(), weights.len());
        let len = spectrum.len().min(weights.len());
        let mut out = vec![Complex64::new(0.0, 0.0); len];
        let chunks = len / LANES;
        for idx in 0..chunks {
            let start = idx * LANES;
            let re = Simd::<f64, LANES>::from_array([
                spectrum[start].re,
                spectrum[start + 1].re,
                spectrum[start + 2].re,
                spectrum[start + 3].re,
            ]);
            let im = Simd::<f64, LANES>::from_array([
                spectrum[start].im,
                spectrum[start + 1].im,
                spectrum[start + 2].im,
                spectrum[start + 3].im,
            ]);
            let w = Simd::<f64, LANES>::from_slice(&weights[start..start + LANES]);
            let re = (re * w).to_array();
            let im = (im * w).to_array();
            for lane in 0..LANES {
                out[start + lane] = Complex64::new(re[lane], im[lane]);
            }
        }
        for idx in chunks * LANES..len {
            out[idx] = spectrum[idx] * weights[idx];
        }
        out
    }

    #[inline]
    pub(crate) fn complex_norms(values: &[Complex64], scale: f64) -> Vec<f64> {
        let mut out = vec![0.0; values.len()];
        write_complex_norms(values, scale, &mut out);
        out
    }

    #[inline]
    pub(crate) fn write_complex_norms(values: &[Complex64], scale: f64, out: &mut [f64]) {
        debug_assert!(out.len() >= values.len());
        let scale = Simd::<f64, LANES>::splat(scale);
        let chunks = values.len() / LANES;
        for idx in 0..chunks {
            let start = idx * LANES;
            let re = Simd::<f64, LANES>::from_array([
                values[start].re,
                values[start + 1].re,
                values[start + 2].re,
                values[start + 3].re,
            ]);
            let im = Simd::<f64, LANES>::from_array([
                values[start].im,
                values[start + 1].im,
                values[start + 2].im,
                values[start + 3].im,
            ]);
            let norms = ((re * re + im * im).sqrt() * scale).to_array();
            out[start..start + LANES].copy_from_slice(&norms);
        }
        for idx in chunks * LANES..values.len() {
            out[idx] = values[idx].norm() * scale[0];
        }
    }
}

#[cfg(not(feature = "portable-simd"))]
mod imp {
    use num_complex::Complex64;

    #[inline]
    pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
        debug_assert_eq!(a.len(), b.len());
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    #[inline]
    pub(crate) fn sum_squares_scaled(values: &[f64], scale: f64) -> f64 {
        values.iter().map(|value| value * value).sum::<f64>() * scale
    }

    #[inline]
    pub(crate) fn sum_squared_diff(a: &[f64], b: &[f64]) -> f64 {
        debug_assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum()
    }

    #[inline]
    pub(crate) fn sum_squared_offset(values: &[f64], offset: f64) -> f64 {
        values
            .iter()
            .map(|value| {
                let d = value - offset;
                d * d
            })
            .sum()
    }

    #[inline]
    pub(crate) fn mul_assign(values: &mut [f64], weights: &[f64]) {
        debug_assert_eq!(values.len(), weights.len());
        for (value, weight) in values.iter_mut().zip(weights.iter()) {
            *value *= *weight;
        }
    }

    #[inline]
    pub(crate) fn scale_assign(values: &mut [f64], scale: f64) {
        for value in values {
            *value *= scale;
        }
    }

    #[inline]
    pub(crate) fn pre_emphasis(sig: &[f64], coeff: f64) -> Vec<f64> {
        if sig.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(sig.len());
        out.push(sig[0]);
        out.extend(sig.windows(2).map(|pair| pair[1] - coeff * pair[0]));
        out
    }

    #[inline]
    pub(crate) fn complex_mul_real(spectrum: &[Complex64], weights: &[f64]) -> Vec<Complex64> {
        debug_assert_eq!(spectrum.len(), weights.len());
        spectrum
            .iter()
            .zip(weights.iter())
            .map(|(bin, weight)| *bin * *weight)
            .collect()
    }

    #[inline]
    pub(crate) fn complex_norms(values: &[Complex64], scale: f64) -> Vec<f64> {
        values.iter().map(|value| value.norm() * scale).collect()
    }

    #[inline]
    pub(crate) fn write_complex_norms(values: &[Complex64], scale: f64, out: &mut [f64]) {
        debug_assert!(out.len() >= values.len());
        for (dst, value) in out.iter_mut().zip(values.iter()) {
            *dst = value.norm() * scale;
        }
    }
}

pub(crate) use imp::*;
