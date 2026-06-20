use crate::{
    BarkConversionApproach, CqccOptions, ErbConversionApproach, FeatureOptions, FilterBankOptions,
    Matrix, MelConversionApproach, Normalization, Scale, SlidingWindow, SpafeError, Vector,
    WindowType,
};
use crate::{cochleagram as coch, fbanks, features, frequencies, spfeats, utils};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type YinOutput = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

fn spafe_err(err: SpafeError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn matrix_to_vec(matrix: Matrix) -> Vec<Vec<f64>> {
    matrix
        .rows()
        .into_iter()
        .map(|row| row.iter().copied().collect())
        .collect()
}

fn vector_to_vec(vector: Vector) -> Vec<f64> {
    vector.to_vec()
}

fn parse_scale(value: &str) -> PyResult<Scale> {
    match value.to_ascii_lowercase().as_str() {
        "ascendant" | "asc" => Ok(Scale::Ascendant),
        "descendant" | "desc" => Ok(Scale::Descendant),
        "constant" | "const" => Ok(Scale::Constant),
        _ => Err(PyValueError::new_err(
            "scale must be 'constant', 'ascendant', or 'descendant'",
        )),
    }
}

fn parse_window_type(value: &str) -> PyResult<WindowType> {
    match value.to_ascii_lowercase().as_str() {
        "hanning" | "hann" => Ok(WindowType::Hanning),
        "bartlet" | "bartlett" => Ok(WindowType::Bartlet),
        "kaiser" => Ok(WindowType::Kaiser),
        "blackman" => Ok(WindowType::Blackman),
        "hamming" => Ok(WindowType::Hamming),
        _ => Err(PyValueError::new_err(
            "win_type must be 'hamming', 'hanning', 'bartlet', 'kaiser', or 'blackman'",
        )),
    }
}

fn parse_normalization(value: Option<&str>) -> PyResult<Option<Normalization>> {
    match value.map(str::to_ascii_lowercase).as_deref() {
        None | Some("none") => Ok(None),
        Some("mean_variance" | "mvn") => Ok(Some(Normalization::MeanVariance)),
        Some("mean_subtraction" | "ms") => Ok(Some(Normalization::MeanSubtraction)),
        Some("variance" | "var") => Ok(Some(Normalization::Variance)),
        Some("mean") => Ok(Some(Normalization::Mean)),
        _ => Err(PyValueError::new_err(
            "normalize must be None, 'mean_variance', 'mean_subtraction', 'variance', or 'mean'",
        )),
    }
}

fn parse_mel(value: &str) -> PyResult<MelConversionApproach> {
    match value.to_ascii_lowercase().as_str() {
        "oshaghnessy" | "oshaughnessy" | "slaney" => Ok(MelConversionApproach::Oshaghnessy),
        "lindsay" => Ok(MelConversionApproach::Lindsay),
        _ => Err(PyValueError::new_err(
            "mel conversion must be 'oshaghnessy' or 'lindsay'",
        )),
    }
}

fn parse_bark(value: &str) -> PyResult<BarkConversionApproach> {
    match value.to_ascii_lowercase().as_str() {
        "wang" => Ok(BarkConversionApproach::Wang),
        "tjomov" => Ok(BarkConversionApproach::Tjomov),
        "schroeder" => Ok(BarkConversionApproach::Schroeder),
        "terhardt" => Ok(BarkConversionApproach::Terhardt),
        "zwicker" => Ok(BarkConversionApproach::Zwicker),
        "traunmueller" | "traunmuller" => Ok(BarkConversionApproach::Traunmueller),
        _ => Err(PyValueError::new_err(
            "bark conversion must be 'wang', 'tjomov', 'schroeder', 'terhardt', 'zwicker', or 'traunmueller'",
        )),
    }
}

fn parse_erb(value: &str) -> PyResult<ErbConversionApproach> {
    match value.to_ascii_lowercase().as_str() {
        "glasberg" => Ok(ErbConversionApproach::Glasberg),
        _ => Err(PyValueError::new_err("erb conversion must be 'glasberg'")),
    }
}

#[pyclass(name = "FilterBankOptions", module = "spafe", from_py_object)]
#[derive(Clone)]
struct PyFilterBankOptions {
    #[pyo3(get, set)]
    nfilts: usize,
    #[pyo3(get, set)]
    nfft: usize,
    #[pyo3(get, set)]
    fs: usize,
    #[pyo3(get, set)]
    low_freq: f64,
    #[pyo3(get, set)]
    high_freq: Option<f64>,
    #[pyo3(get, set)]
    scale: String,
}

impl PyFilterBankOptions {
    fn to_rust(&self) -> PyResult<FilterBankOptions> {
        Ok(FilterBankOptions {
            nfilts: self.nfilts,
            nfft: self.nfft,
            fs: self.fs,
            low_freq: self.low_freq,
            high_freq: self.high_freq,
            scale: parse_scale(&self.scale)?,
        })
    }
}

#[pymethods]
impl PyFilterBankOptions {
    #[new]
    #[pyo3(signature = (nfilts=24, nfft=512, fs=16_000, low_freq=0.0, high_freq=None, scale="constant".to_string()))]
    fn new(
        nfilts: usize,
        nfft: usize,
        fs: usize,
        low_freq: f64,
        high_freq: Option<f64>,
        scale: String,
    ) -> Self {
        Self {
            nfilts,
            nfft,
            fs,
            low_freq,
            high_freq,
            scale,
        }
    }
}

impl Default for PyFilterBankOptions {
    fn default() -> Self {
        let opts = FilterBankOptions::default();
        Self::new(
            opts.nfilts,
            opts.nfft,
            opts.fs,
            opts.low_freq,
            opts.high_freq,
            "constant".to_string(),
        )
    }
}

#[pyclass(name = "FeatureOptions", module = "spafe", from_py_object)]
#[derive(Clone)]
struct PyFeatureOptions {
    #[pyo3(get, set)]
    fs: usize,
    #[pyo3(get, set)]
    num_ceps: usize,
    #[pyo3(get, set)]
    pre_emph: bool,
    #[pyo3(get, set)]
    pre_emph_coeff: f64,
    #[pyo3(get, set)]
    win_len: f64,
    #[pyo3(get, set)]
    win_hop: f64,
    #[pyo3(get, set)]
    win_type: String,
    #[pyo3(get, set)]
    nfilts: usize,
    #[pyo3(get, set)]
    nfft: usize,
    #[pyo3(get, set)]
    low_freq: f64,
    #[pyo3(get, set)]
    high_freq: Option<f64>,
    #[pyo3(get, set)]
    scale: String,
    #[pyo3(get, set)]
    dct_type: usize,
    #[pyo3(get, set)]
    use_energy: bool,
    #[pyo3(get, set)]
    lifter: Option<i32>,
    #[pyo3(get, set)]
    normalize: Option<String>,
}

impl PyFeatureOptions {
    fn to_rust(&self) -> PyResult<FeatureOptions> {
        Ok(FeatureOptions {
            fs: self.fs,
            num_ceps: self.num_ceps,
            pre_emph: self.pre_emph,
            pre_emph_coeff: self.pre_emph_coeff,
            window: SlidingWindow {
                win_len: self.win_len,
                win_hop: self.win_hop,
                win_type: parse_window_type(&self.win_type)?,
            },
            nfilts: self.nfilts,
            nfft: self.nfft,
            low_freq: self.low_freq,
            high_freq: self.high_freq,
            scale: parse_scale(&self.scale)?,
            dct_type: self.dct_type,
            use_energy: self.use_energy,
            lifter: self.lifter,
            normalize: parse_normalization(self.normalize.as_deref())?,
        })
    }
}

#[pymethods]
impl PyFeatureOptions {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        fs=16_000,
        num_ceps=13,
        pre_emph=true,
        pre_emph_coeff=0.97,
        win_len=0.025,
        win_hop=0.010,
        win_type="hamming".to_string(),
        nfilts=24,
        nfft=512,
        low_freq=0.0,
        high_freq=None,
        scale="constant".to_string(),
        dct_type=2,
        use_energy=false,
        lifter=None,
        normalize=None
    ))]
    fn new(
        fs: usize,
        num_ceps: usize,
        pre_emph: bool,
        pre_emph_coeff: f64,
        win_len: f64,
        win_hop: f64,
        win_type: String,
        nfilts: usize,
        nfft: usize,
        low_freq: f64,
        high_freq: Option<f64>,
        scale: String,
        dct_type: usize,
        use_energy: bool,
        lifter: Option<i32>,
        normalize: Option<String>,
    ) -> Self {
        Self {
            fs,
            num_ceps,
            pre_emph,
            pre_emph_coeff,
            win_len,
            win_hop,
            win_type,
            nfilts,
            nfft,
            low_freq,
            high_freq,
            scale,
            dct_type,
            use_energy,
            lifter,
            normalize,
        }
    }
}

impl Default for PyFeatureOptions {
    fn default() -> Self {
        let opts = FeatureOptions::default();
        Self::new(
            opts.fs,
            opts.num_ceps,
            opts.pre_emph,
            opts.pre_emph_coeff,
            opts.window.win_len,
            opts.window.win_hop,
            "hamming".to_string(),
            opts.nfilts,
            opts.nfft,
            opts.low_freq,
            opts.high_freq,
            "constant".to_string(),
            opts.dct_type,
            opts.use_energy,
            opts.lifter,
            None,
        )
    }
}

#[pyclass(name = "CqccOptions", module = "spafe", from_py_object)]
#[derive(Clone)]
struct PyCqccOptions {
    #[pyo3(get, set)]
    number_of_octaves: usize,
    #[pyo3(get, set)]
    number_of_bins_per_octave: usize,
    #[pyo3(get, set)]
    resampling_ratio: f64,
    #[pyo3(get, set)]
    spectral_threshold: f64,
    #[pyo3(get, set)]
    f0: f64,
    #[pyo3(get, set)]
    q_rate: f64,
}

impl PyCqccOptions {
    fn to_rust(&self) -> CqccOptions {
        CqccOptions {
            number_of_octaves: self.number_of_octaves,
            number_of_bins_per_octave: self.number_of_bins_per_octave,
            resampling_ratio: self.resampling_ratio,
            spectral_threshold: self.spectral_threshold,
            f0: self.f0,
            q_rate: self.q_rate,
        }
    }
}

#[pymethods]
impl PyCqccOptions {
    #[new]
    #[pyo3(signature = (
        number_of_octaves=7,
        number_of_bins_per_octave=24,
        resampling_ratio=1.0,
        spectral_threshold=0.005,
        f0=120.0,
        q_rate=1.0
    ))]
    fn new(
        number_of_octaves: usize,
        number_of_bins_per_octave: usize,
        resampling_ratio: f64,
        spectral_threshold: f64,
        f0: f64,
        q_rate: f64,
    ) -> Self {
        Self {
            number_of_octaves,
            number_of_bins_per_octave,
            resampling_ratio,
            spectral_threshold,
            f0,
            q_rate,
        }
    }
}

impl Default for PyCqccOptions {
    fn default() -> Self {
        let opts = CqccOptions::default();
        Self::new(
            opts.number_of_octaves,
            opts.number_of_bins_per_octave,
            opts.resampling_ratio,
            opts.spectral_threshold,
            opts.f0,
            opts.q_rate,
        )
    }
}

#[pyclass(name = "CochleagramOptions", module = "spafe", from_py_object)]
#[derive(Clone)]
struct PyCochleagramOptions {
    #[pyo3(get, set)]
    signal_size: usize,
    #[pyo3(get, set)]
    sr: usize,
    #[pyo3(get, set)]
    env_sr: usize,
    #[pyo3(get, set)]
    pad_factor: Option<usize>,
    #[pyo3(get, set)]
    filter_n: usize,
    #[pyo3(get, set)]
    low_lim: f64,
    #[pyo3(get, set)]
    high_lim: f64,
    #[pyo3(get, set)]
    sample_factor: usize,
    #[pyo3(get, set)]
    envelope: String,
    #[pyo3(get, set)]
    downsampling: String,
    #[pyo3(get, set)]
    downsampling_window_size: usize,
    #[pyo3(get, set)]
    compression: Option<String>,
    #[pyo3(get, set)]
    compression_scale: f64,
    #[pyo3(get, set)]
    compression_offset: f64,
    #[pyo3(get, set)]
    compression_power: f64,
    #[pyo3(get, set)]
    downsample_then_compress: bool,
}

impl PyCochleagramOptions {
    fn to_rust(&self) -> PyResult<coch::CochleagramOptions> {
        let envelope = match self.envelope.to_ascii_lowercase().as_str() {
            "hilbert" => coch::EnvelopeMode::Hilbert,
            "abs_subbands" | "abs" => coch::EnvelopeMode::AbsSubbands,
            "rectify_subbands" | "rectify" => coch::EnvelopeMode::RectifySubbands,
            _ => {
                return Err(PyValueError::new_err(
                    "envelope must be 'hilbert', 'abs_subbands', or 'rectify_subbands'",
                ));
            }
        };
        let downsampling = match self.downsampling.to_ascii_lowercase().as_str() {
            "sinc" | "sinc_with_kaiser_window" => coch::DownsamplingMode::SincWithKaiserWindow {
                window_size: self.downsampling_window_size,
                padding: None,
            },
            "hann" | "hann_pooling1d" => coch::DownsamplingMode::HannPooling1d {
                window_size: self.downsampling_window_size,
                padding: 0,
                normalize: true,
            },
            _ => {
                return Err(PyValueError::new_err(
                    "downsampling must be 'sinc' or 'hann'",
                ));
            }
        };
        let compression = match self.compression.as_deref().map(str::to_ascii_lowercase) {
            None => None,
            Some(value) if value == "none" => None,
            Some(value) if value == "linear" => Some(coch::CompressionMode::Linear {
                scale: self.compression_scale,
                offset: self.compression_offset,
            }),
            Some(value) if value == "power" => Some(coch::CompressionMode::Power {
                scale: self.compression_scale,
                offset: self.compression_offset,
                power: self.compression_power,
            }),
            Some(value) if value == "clipped_grad_power" => {
                Some(coch::CompressionMode::ClippedGradPower {
                    scale: self.compression_scale,
                    offset: self.compression_offset,
                    power: self.compression_power,
                    clip_value: 5.0,
                })
            }
            _ => {
                return Err(PyValueError::new_err(
                    "compression must be None, 'linear', 'power', or 'clipped_grad_power'",
                ));
            }
        };
        Ok(coch::CochleagramOptions {
            signal_size: self.signal_size,
            sr: self.sr,
            env_sr: self.env_sr,
            pad_factor: self.pad_factor,
            use_rfft: true,
            filter: coch::ErbCosFilterOptions {
                n: self.filter_n,
                low_lim: self.low_lim,
                high_lim: self.high_lim,
                sample_factor: self.sample_factor,
                full_filter: false,
                no_lowpass: false,
                no_highpass: false,
                strict: true,
            },
            envelope,
            downsampling,
            compression,
            downsample_then_compress: self.downsample_then_compress,
        })
    }
}

#[pymethods]
impl PyCochleagramOptions {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        signal_size=40_000,
        sr=20_000,
        env_sr=200,
        pad_factor=None,
        filter_n=50,
        low_lim=50.0,
        high_lim=10_000.0,
        sample_factor=4,
        envelope="hilbert".to_string(),
        downsampling="sinc".to_string(),
        downsampling_window_size=1001,
        compression=Some("clipped_grad_power".to_string()),
        compression_scale=1.0,
        compression_offset=1e-8,
        compression_power=0.3,
        downsample_then_compress=true
    ))]
    fn new(
        signal_size: usize,
        sr: usize,
        env_sr: usize,
        pad_factor: Option<usize>,
        filter_n: usize,
        low_lim: f64,
        high_lim: f64,
        sample_factor: usize,
        envelope: String,
        downsampling: String,
        downsampling_window_size: usize,
        compression: Option<String>,
        compression_scale: f64,
        compression_offset: f64,
        compression_power: f64,
        downsample_then_compress: bool,
    ) -> Self {
        Self {
            signal_size,
            sr,
            env_sr,
            pad_factor,
            filter_n,
            low_lim,
            high_lim,
            sample_factor,
            envelope,
            downsampling,
            downsampling_window_size,
            compression,
            compression_scale,
            compression_offset,
            compression_power,
            downsample_then_compress,
        }
    }
}

#[pyclass(name = "SpectrogramOutput", module = "spafe")]
struct PySpectrogramOutput {
    #[pyo3(get)]
    features: Vec<Vec<f64>>,
    #[pyo3(get)]
    fft_magnitude: Vec<Vec<f64>>,
}

#[pyclass(name = "SpectralFeats", module = "spafe")]
struct PySpectralFeats {
    #[pyo3(get)]
    spectral_centroid: f64,
    #[pyo3(get)]
    spectral_skewness: f64,
    #[pyo3(get)]
    spectral_kurtosis: f64,
    #[pyo3(get)]
    spectral_entropy: f64,
    #[pyo3(get)]
    spectral_spread: Vec<f64>,
    #[pyo3(get)]
    spectral_flatness: f64,
    #[pyo3(get)]
    spectral_flux: f64,
    #[pyo3(get)]
    spectral_mean: (f64, f64),
    #[pyo3(get)]
    spectral_rms: (f64, f64),
    #[pyo3(get)]
    spectral_std: f64,
    #[pyo3(get)]
    spectral_variance: f64,
}

#[pyclass(name = "CochleagramOutput", module = "spafe")]
struct PyCochleagramOutput {
    #[pyo3(get)]
    cochleagram: Vec<Vec<f64>>,
    #[pyo3(get)]
    envelopes: Vec<Vec<f64>>,
    #[pyo3(get)]
    downsampled: Vec<Vec<f64>>,
    #[pyo3(get)]
    center_freqs: Vec<f64>,
    #[pyo3(get)]
    freqs: Vec<f64>,
}

fn feature_opts(opts: Option<&PyFeatureOptions>) -> PyResult<FeatureOptions> {
    match opts {
        Some(opts) => opts.to_rust(),
        None => PyFeatureOptions::default().to_rust(),
    }
}

fn filter_opts(opts: Option<&PyFilterBankOptions>) -> PyResult<FilterBankOptions> {
    match opts {
        Some(opts) => opts.to_rust(),
        None => PyFilterBankOptions::default().to_rust(),
    }
}

fn cqcc_options(opts: Option<&PyCqccOptions>) -> CqccOptions {
    match opts {
        Some(opts) => opts.to_rust(),
        None => PyCqccOptions::default().to_rust(),
    }
}

#[pyfunction]
#[pyo3(signature = (opts=None, conversion="oshaghnessy"))]
fn mel_filter_banks(
    opts: Option<&PyFilterBankOptions>,
    conversion: &str,
) -> PyResult<(Vec<Vec<f64>>, Vec<f64>)> {
    let (matrix, centers) =
        fbanks::mel_filter_banks(&filter_opts(opts)?, parse_mel(conversion)?).map_err(spafe_err)?;
    Ok((matrix_to_vec(matrix), vector_to_vec(centers)))
}

#[pyfunction]
#[pyo3(signature = (opts=None, conversion="oshaghnessy"))]
fn inverse_mel_filter_banks(
    opts: Option<&PyFilterBankOptions>,
    conversion: &str,
) -> PyResult<(Vec<Vec<f64>>, Vec<f64>)> {
    let (matrix, centers) =
        fbanks::inverse_mel_filter_banks(&filter_opts(opts)?, parse_mel(conversion)?)
            .map_err(spafe_err)?;
    Ok((matrix_to_vec(matrix), vector_to_vec(centers)))
}

#[pyfunction]
#[pyo3(signature = (opts=None))]
fn linear_filter_banks(opts: Option<&PyFilterBankOptions>) -> PyResult<(Vec<Vec<f64>>, Vec<f64>)> {
    let (matrix, centers) = fbanks::linear_filter_banks(&filter_opts(opts)?).map_err(spafe_err)?;
    Ok((matrix_to_vec(matrix), vector_to_vec(centers)))
}

#[pyfunction]
#[pyo3(signature = (opts=None, conversion="wang"))]
fn bark_filter_banks(
    opts: Option<&PyFilterBankOptions>,
    conversion: &str,
) -> PyResult<(Vec<Vec<f64>>, Vec<f64>)> {
    let (matrix, centers) = fbanks::bark_filter_banks(&filter_opts(opts)?, parse_bark(conversion)?)
        .map_err(spafe_err)?;
    Ok((matrix_to_vec(matrix), vector_to_vec(centers)))
}

#[pyfunction]
#[pyo3(signature = (opts=None, order=4, conversion="glasberg"))]
fn gammatone_filter_banks(
    opts: Option<&PyFilterBankOptions>,
    order: i32,
    conversion: &str,
) -> PyResult<(Vec<Vec<f64>>, Vec<f64>)> {
    let (matrix, centers) =
        fbanks::gammatone_filter_banks(&filter_opts(opts)?, order, parse_erb(conversion)?)
            .map_err(spafe_err)?;
    Ok((matrix_to_vec(matrix), vector_to_vec(centers)))
}

fn spectrogram_output(output: crate::SpectrogramOutput) -> PySpectrogramOutput {
    PySpectrogramOutput {
        features: matrix_to_vec(output.features),
        fft_magnitude: matrix_to_vec(output.fft_magnitude),
    }
}

#[pyfunction]
#[pyo3(signature = (sig, opts=None))]
fn mel_spectrogram(
    sig: Vec<f64>,
    opts: Option<&PyFeatureOptions>,
) -> PyResult<PySpectrogramOutput> {
    features::mel_spectrogram(&sig, &feature_opts(opts)?)
        .map(spectrogram_output)
        .map_err(spafe_err)
}

#[pyfunction]
#[pyo3(signature = (sig, opts=None))]
fn linear_spectrogram(
    sig: Vec<f64>,
    opts: Option<&PyFeatureOptions>,
) -> PyResult<PySpectrogramOutput> {
    features::linear_spectrogram(&sig, &feature_opts(opts)?)
        .map(spectrogram_output)
        .map_err(spafe_err)
}

#[pyfunction]
#[pyo3(signature = (sig, opts=None))]
fn bark_spectrogram(
    sig: Vec<f64>,
    opts: Option<&PyFeatureOptions>,
) -> PyResult<PySpectrogramOutput> {
    features::bark_spectrogram(&sig, &feature_opts(opts)?)
        .map(spectrogram_output)
        .map_err(spafe_err)
}

#[pyfunction]
#[pyo3(signature = (sig, opts=None))]
fn erb_spectrogram(
    sig: Vec<f64>,
    opts: Option<&PyFeatureOptions>,
) -> PyResult<PySpectrogramOutput> {
    features::erb_spectrogram(&sig, &feature_opts(opts)?)
        .map(spectrogram_output)
        .map_err(spafe_err)
}

#[pyfunction]
#[pyo3(signature = (sig, opts=None))]
fn mfcc(sig: Vec<f64>, opts: Option<&PyFeatureOptions>) -> PyResult<Vec<Vec<f64>>> {
    features::mfcc(&sig, &feature_opts(opts)?)
        .map(matrix_to_vec)
        .map_err(spafe_err)
}

#[pyfunction]
#[pyo3(signature = (sig, opts=None))]
fn imfcc(sig: Vec<f64>, opts: Option<&PyFeatureOptions>) -> PyResult<Vec<Vec<f64>>> {
    features::imfcc(&sig, &feature_opts(opts)?)
        .map(matrix_to_vec)
        .map_err(spafe_err)
}

#[pyfunction]
#[pyo3(signature = (sig, opts=None))]
fn lfcc(sig: Vec<f64>, opts: Option<&PyFeatureOptions>) -> PyResult<Vec<Vec<f64>>> {
    features::lfcc(&sig, &feature_opts(opts)?)
        .map(matrix_to_vec)
        .map_err(spafe_err)
}

#[pyfunction]
#[pyo3(signature = (sig, opts=None))]
fn bfcc(sig: Vec<f64>, opts: Option<&PyFeatureOptions>) -> PyResult<Vec<Vec<f64>>> {
    features::bfcc(&sig, &feature_opts(opts)?)
        .map(matrix_to_vec)
        .map_err(spafe_err)
}

#[pyfunction]
#[pyo3(signature = (sig, opts=None))]
fn gfcc(sig: Vec<f64>, opts: Option<&PyFeatureOptions>) -> PyResult<Vec<Vec<f64>>> {
    features::gfcc(&sig, &feature_opts(opts)?)
        .map(matrix_to_vec)
        .map_err(spafe_err)
}

#[pyfunction]
#[pyo3(signature = (sig, opts=None, gamma=-1.0 / 7.0))]
fn msrcc(sig: Vec<f64>, opts: Option<&PyFeatureOptions>, gamma: f64) -> PyResult<Vec<Vec<f64>>> {
    features::msrcc(&sig, &feature_opts(opts)?, gamma)
        .map(matrix_to_vec)
        .map_err(spafe_err)
}

#[pyfunction]
#[pyo3(signature = (sig, opts=None))]
fn ngcc(sig: Vec<f64>, opts: Option<&PyFeatureOptions>) -> PyResult<Vec<Vec<f64>>> {
    features::ngcc(&sig, &feature_opts(opts)?)
        .map(matrix_to_vec)
        .map_err(spafe_err)
}

#[pyfunction]
#[pyo3(signature = (sig, opts=None, gamma=-1.0 / 7.0))]
fn psrcc(sig: Vec<f64>, opts: Option<&PyFeatureOptions>, gamma: f64) -> PyResult<Vec<Vec<f64>>> {
    features::psrcc(&sig, &feature_opts(opts)?, gamma)
        .map(matrix_to_vec)
        .map_err(spafe_err)
}

#[pyfunction]
#[pyo3(signature = (sig, opts=None, power=2.0))]
fn pncc(sig: Vec<f64>, opts: Option<&PyFeatureOptions>, power: f64) -> PyResult<Vec<Vec<f64>>> {
    features::pncc(&sig, &feature_opts(opts)?, power)
        .map(matrix_to_vec)
        .map_err(spafe_err)
}

#[pyfunction]
#[pyo3(signature = (sig, opts=None))]
fn lpcc(sig: Vec<f64>, opts: Option<&PyFeatureOptions>) -> PyResult<Vec<Vec<f64>>> {
    features::lpcc(&sig, &feature_opts(opts)?)
        .map(matrix_to_vec)
        .map_err(spafe_err)
}

#[pyfunction]
#[pyo3(signature = (sig, opts=None, do_rasta=false))]
fn plp(sig: Vec<f64>, opts: Option<&PyFeatureOptions>, do_rasta: bool) -> PyResult<Vec<Vec<f64>>> {
    features::plp(&sig, &feature_opts(opts)?, do_rasta)
        .map(matrix_to_vec)
        .map_err(spafe_err)
}

#[pyfunction]
#[pyo3(signature = (sig, opts=None))]
fn rplp(sig: Vec<f64>, opts: Option<&PyFeatureOptions>) -> PyResult<Vec<Vec<f64>>> {
    features::rplp(&sig, &feature_opts(opts)?)
        .map(matrix_to_vec)
        .map_err(spafe_err)
}

#[pyfunction]
#[pyo3(signature = (sig, opts=None, cqcc_opts=None))]
fn cqcc(
    sig: Vec<f64>,
    opts: Option<&PyFeatureOptions>,
    cqcc_opts: Option<&PyCqccOptions>,
) -> PyResult<Vec<Vec<f64>>> {
    features::cqcc(&sig, &feature_opts(opts)?, &cqcc_options(cqcc_opts))
        .map(matrix_to_vec)
        .map_err(spafe_err)
}

#[pyfunction]
#[pyo3(signature = (sig, fs=16_000, win_len=0.03, win_hop=0.015, low_freq=50.0, high_freq=1000.0, harmonic_threshold=0.1))]
fn compute_yin(
    sig: Vec<f64>,
    fs: usize,
    win_len: f64,
    win_hop: f64,
    low_freq: f64,
    high_freq: f64,
    harmonic_threshold: f64,
) -> PyResult<YinOutput> {
    let (pitches, harmonic_rates, argmins, times) = frequencies::compute_yin(
        &sig,
        fs,
        win_len,
        win_hop,
        low_freq,
        high_freq,
        harmonic_threshold,
    )
    .map_err(spafe_err)?;
    Ok((
        vector_to_vec(pitches),
        vector_to_vec(harmonic_rates),
        vector_to_vec(argmins),
        vector_to_vec(times),
    ))
}

#[pyfunction]
#[pyo3(signature = (sig, fs=16_000, nfft=512, win_len=0.025, win_hop=0.010, win_type="hamming", only_positive=true))]
fn get_dominant_frequencies(
    sig: Vec<f64>,
    fs: usize,
    nfft: usize,
    win_len: f64,
    win_hop: f64,
    win_type: &str,
    only_positive: bool,
) -> PyResult<Vec<f64>> {
    frequencies::get_dominant_frequencies(
        &sig,
        fs,
        nfft,
        win_len,
        win_hop,
        parse_window_type(win_type)?,
        only_positive,
    )
    .map(vector_to_vec)
    .map_err(spafe_err)
}

#[pyfunction(name = "cochleagram")]
#[pyo3(signature = (sig, opts=None))]
fn py_cochleagram(
    sig: Vec<f64>,
    opts: Option<&PyCochleagramOptions>,
) -> PyResult<PyCochleagramOutput> {
    let opts = match opts {
        Some(opts) => opts.to_rust()?,
        None => PyCochleagramOptions::new(
            sig.len(),
            20_000,
            200,
            None,
            50,
            50.0,
            10_000.0,
            4,
            "hilbert".to_string(),
            "sinc".to_string(),
            1001,
            Some("clipped_grad_power".to_string()),
            1.0,
            1e-8,
            0.3,
            true,
        )
        .to_rust()?,
    };
    let output = coch::cochleagram(&sig, &opts).map_err(spafe_err)?;
    Ok(PyCochleagramOutput {
        cochleagram: matrix_to_vec(output.cochleagram),
        envelopes: matrix_to_vec(output.latents.envelopes),
        downsampled: matrix_to_vec(output.latents.downsampled),
        center_freqs: output.filter_bank.center_freqs,
        freqs: output.filter_bank.freqs,
    })
}

#[pyfunction]
#[pyo3(signature = (freq, conversion="oshaghnessy"))]
fn hz2mel(freq: f64, conversion: &str) -> PyResult<f64> {
    Ok(utils::converters::hz2mel(freq, parse_mel(conversion)?))
}

#[pyfunction]
#[pyo3(signature = (freq, conversion="oshaghnessy"))]
fn mel2hz(freq: f64, conversion: &str) -> PyResult<f64> {
    Ok(utils::converters::mel2hz(freq, parse_mel(conversion)?))
}

#[pyfunction]
#[pyo3(signature = (freq, conversion="wang"))]
fn hz2bark(freq: f64, conversion: &str) -> PyResult<f64> {
    Ok(utils::converters::hz2bark(freq, parse_bark(conversion)?))
}

#[pyfunction]
#[pyo3(signature = (freq, conversion="wang"))]
fn bark2hz(freq: f64, conversion: &str) -> PyResult<f64> {
    Ok(utils::converters::bark2hz(freq, parse_bark(conversion)?))
}

#[pyfunction]
#[pyo3(signature = (freq, conversion="glasberg"))]
fn hz2erb(freq: f64, conversion: &str) -> PyResult<f64> {
    Ok(utils::converters::hz2erb(freq, parse_erb(conversion)?))
}

#[pyfunction]
#[pyo3(signature = (freq, conversion="glasberg"))]
fn erb2hz(freq: f64, conversion: &str) -> PyResult<f64> {
    Ok(utils::converters::erb2hz(freq, parse_erb(conversion)?))
}

#[pyfunction]
#[pyo3(signature = (sig, fs, nfft=512))]
fn extract_feats(sig: Vec<f64>, fs: usize, nfft: usize) -> PySpectralFeats {
    let feats = spfeats::extract_feats_with_nfft(&sig, fs, nfft);
    PySpectralFeats {
        spectral_centroid: feats.spectral_centroid,
        spectral_skewness: feats.spectral_skewness,
        spectral_kurtosis: feats.spectral_kurtosis,
        spectral_entropy: feats.spectral_entropy,
        spectral_spread: feats.spectral_spread,
        spectral_flatness: feats.spectral_flatness,
        spectral_flux: feats.spectral_flux,
        spectral_mean: (feats.spectral_mean.re, feats.spectral_mean.im),
        spectral_rms: (feats.spectral_rms.re, feats.spectral_rms.im),
        spectral_std: feats.spectral_std,
        spectral_variance: feats.spectral_variance,
    }
}

#[pymodule]
fn spafe(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyFilterBankOptions>()?;
    module.add_class::<PyFeatureOptions>()?;
    module.add_class::<PyCqccOptions>()?;
    module.add_class::<PyCochleagramOptions>()?;
    module.add_class::<PySpectrogramOutput>()?;
    module.add_class::<PySpectralFeats>()?;
    module.add_class::<PyCochleagramOutput>()?;

    module.add_function(wrap_pyfunction!(mel_filter_banks, module)?)?;
    module.add_function(wrap_pyfunction!(inverse_mel_filter_banks, module)?)?;
    module.add_function(wrap_pyfunction!(linear_filter_banks, module)?)?;
    module.add_function(wrap_pyfunction!(bark_filter_banks, module)?)?;
    module.add_function(wrap_pyfunction!(gammatone_filter_banks, module)?)?;

    module.add_function(wrap_pyfunction!(mel_spectrogram, module)?)?;
    module.add_function(wrap_pyfunction!(linear_spectrogram, module)?)?;
    module.add_function(wrap_pyfunction!(bark_spectrogram, module)?)?;
    module.add_function(wrap_pyfunction!(erb_spectrogram, module)?)?;
    module.add_function(wrap_pyfunction!(mfcc, module)?)?;
    module.add_function(wrap_pyfunction!(imfcc, module)?)?;
    module.add_function(wrap_pyfunction!(lfcc, module)?)?;
    module.add_function(wrap_pyfunction!(bfcc, module)?)?;
    module.add_function(wrap_pyfunction!(gfcc, module)?)?;
    module.add_function(wrap_pyfunction!(msrcc, module)?)?;
    module.add_function(wrap_pyfunction!(ngcc, module)?)?;
    module.add_function(wrap_pyfunction!(psrcc, module)?)?;
    module.add_function(wrap_pyfunction!(pncc, module)?)?;
    module.add_function(wrap_pyfunction!(lpcc, module)?)?;
    module.add_function(wrap_pyfunction!(plp, module)?)?;
    module.add_function(wrap_pyfunction!(rplp, module)?)?;
    module.add_function(wrap_pyfunction!(cqcc, module)?)?;

    module.add_function(wrap_pyfunction!(compute_yin, module)?)?;
    module.add_function(wrap_pyfunction!(get_dominant_frequencies, module)?)?;
    module.add_function(wrap_pyfunction!(py_cochleagram, module)?)?;
    module.add_function(wrap_pyfunction!(extract_feats, module)?)?;

    module.add_function(wrap_pyfunction!(hz2mel, module)?)?;
    module.add_function(wrap_pyfunction!(mel2hz, module)?)?;
    module.add_function(wrap_pyfunction!(hz2bark, module)?)?;
    module.add_function(wrap_pyfunction!(bark2hz, module)?)?;
    module.add_function(wrap_pyfunction!(hz2erb, module)?)?;
    module.add_function(wrap_pyfunction!(erb2hz, module)?)?;

    Ok(())
}
