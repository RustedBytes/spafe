from typing import List, Optional, Sequence, Tuple

Vector = List[float]
Matrix = List[List[float]]
FloatSequence = Sequence[float]
ComplexTuple = Tuple[float, float]


class FilterBankOptions:
    nfilts: int
    nfft: int
    fs: int
    low_freq: float
    high_freq: Optional[float]
    scale: str

    def __init__(
        self,
        nfilts: int = 24,
        nfft: int = 512,
        fs: int = 16_000,
        low_freq: float = 0.0,
        high_freq: Optional[float] = None,
        scale: str = "constant",
    ) -> None: ...


class FeatureOptions:
    fs: int
    num_ceps: int
    pre_emph: bool
    pre_emph_coeff: float
    win_len: float
    win_hop: float
    win_type: str
    nfilts: int
    nfft: int
    low_freq: float
    high_freq: Optional[float]
    scale: str
    dct_type: int
    use_energy: bool
    lifter: Optional[int]
    normalize: Optional[str]

    def __init__(
        self,
        fs: int = 16_000,
        num_ceps: int = 13,
        pre_emph: bool = True,
        pre_emph_coeff: float = 0.97,
        win_len: float = 0.025,
        win_hop: float = 0.010,
        win_type: str = "hamming",
        nfilts: int = 24,
        nfft: int = 512,
        low_freq: float = 0.0,
        high_freq: Optional[float] = None,
        scale: str = "constant",
        dct_type: int = 2,
        use_energy: bool = False,
        lifter: Optional[int] = None,
        normalize: Optional[str] = None,
    ) -> None: ...


class CqccOptions:
    number_of_octaves: int
    number_of_bins_per_octave: int
    resampling_ratio: float
    spectral_threshold: float
    f0: float
    q_rate: float

    def __init__(
        self,
        number_of_octaves: int = 7,
        number_of_bins_per_octave: int = 24,
        resampling_ratio: float = 1.0,
        spectral_threshold: float = 0.005,
        f0: float = 120.0,
        q_rate: float = 1.0,
    ) -> None: ...


class CochleagramOptions:
    signal_size: int
    sr: int
    env_sr: int
    pad_factor: Optional[int]
    filter_n: int
    low_lim: float
    high_lim: float
    sample_factor: int
    envelope: str
    downsampling: str
    downsampling_window_size: int
    compression: Optional[str]
    compression_scale: float
    compression_offset: float
    compression_power: float
    downsample_then_compress: bool

    def __init__(
        self,
        signal_size: int = 40_000,
        sr: int = 20_000,
        env_sr: int = 200,
        pad_factor: Optional[int] = None,
        filter_n: int = 50,
        low_lim: float = 50.0,
        high_lim: float = 10_000.0,
        sample_factor: int = 4,
        envelope: str = "hilbert",
        downsampling: str = "sinc",
        downsampling_window_size: int = 1001,
        compression: Optional[str] = "clipped_grad_power",
        compression_scale: float = 1.0,
        compression_offset: float = 1e-8,
        compression_power: float = 0.3,
        downsample_then_compress: bool = True,
    ) -> None: ...


class SpectrogramOutput:
    @property
    def features(self) -> Matrix: ...

    @property
    def fft_magnitude(self) -> Matrix: ...


class SpectralFeats:
    @property
    def spectral_centroid(self) -> float: ...

    @property
    def spectral_skewness(self) -> float: ...

    @property
    def spectral_kurtosis(self) -> float: ...

    @property
    def spectral_entropy(self) -> float: ...

    @property
    def spectral_spread(self) -> Vector: ...

    @property
    def spectral_flatness(self) -> float: ...

    @property
    def spectral_flux(self) -> float: ...

    @property
    def spectral_mean(self) -> ComplexTuple: ...

    @property
    def spectral_rms(self) -> ComplexTuple: ...

    @property
    def spectral_std(self) -> float: ...

    @property
    def spectral_variance(self) -> float: ...


class CochleagramOutput:
    @property
    def cochleagram(self) -> Matrix: ...

    @property
    def envelopes(self) -> Matrix: ...

    @property
    def downsampled(self) -> Matrix: ...

    @property
    def center_freqs(self) -> Vector: ...

    @property
    def freqs(self) -> Vector: ...


def mel_filter_banks(
    opts: Optional[FilterBankOptions] = None,
    conversion: str = "oshaghnessy",
) -> Tuple[Matrix, Vector]: ...
def inverse_mel_filter_banks(
    opts: Optional[FilterBankOptions] = None,
    conversion: str = "oshaghnessy",
) -> Tuple[Matrix, Vector]: ...
def linear_filter_banks(
    opts: Optional[FilterBankOptions] = None,
) -> Tuple[Matrix, Vector]: ...
def bark_filter_banks(
    opts: Optional[FilterBankOptions] = None,
    conversion: str = "wang",
) -> Tuple[Matrix, Vector]: ...
def gammatone_filter_banks(
    opts: Optional[FilterBankOptions] = None,
    order: int = 4,
    conversion: str = "glasberg",
) -> Tuple[Matrix, Vector]: ...

def mel_spectrogram(
    sig: FloatSequence,
    opts: Optional[FeatureOptions] = None,
) -> SpectrogramOutput: ...
def linear_spectrogram(
    sig: FloatSequence,
    opts: Optional[FeatureOptions] = None,
) -> SpectrogramOutput: ...
def bark_spectrogram(
    sig: FloatSequence,
    opts: Optional[FeatureOptions] = None,
) -> SpectrogramOutput: ...
def erb_spectrogram(
    sig: FloatSequence,
    opts: Optional[FeatureOptions] = None,
) -> SpectrogramOutput: ...

def mfcc(sig: FloatSequence, opts: Optional[FeatureOptions] = None) -> Matrix: ...
def imfcc(sig: FloatSequence, opts: Optional[FeatureOptions] = None) -> Matrix: ...
def lfcc(sig: FloatSequence, opts: Optional[FeatureOptions] = None) -> Matrix: ...
def bfcc(sig: FloatSequence, opts: Optional[FeatureOptions] = None) -> Matrix: ...
def gfcc(sig: FloatSequence, opts: Optional[FeatureOptions] = None) -> Matrix: ...
def msrcc(
    sig: FloatSequence,
    opts: Optional[FeatureOptions] = None,
    gamma: float = -1.0 / 7.0,
) -> Matrix: ...
def ngcc(sig: FloatSequence, opts: Optional[FeatureOptions] = None) -> Matrix: ...
def psrcc(
    sig: FloatSequence,
    opts: Optional[FeatureOptions] = None,
    gamma: float = -1.0 / 7.0,
) -> Matrix: ...
def pncc(
    sig: FloatSequence,
    opts: Optional[FeatureOptions] = None,
    power: float = 2.0,
) -> Matrix: ...
def lpcc(sig: FloatSequence, opts: Optional[FeatureOptions] = None) -> Matrix: ...
def plp(
    sig: FloatSequence,
    opts: Optional[FeatureOptions] = None,
    do_rasta: bool = False,
) -> Matrix: ...
def rplp(sig: FloatSequence, opts: Optional[FeatureOptions] = None) -> Matrix: ...
def cqcc(
    sig: FloatSequence,
    opts: Optional[FeatureOptions] = None,
    cqcc_opts: Optional[CqccOptions] = None,
) -> Matrix: ...

def compute_yin(
    sig: FloatSequence,
    fs: int = 16_000,
    win_len: float = 0.03,
    win_hop: float = 0.015,
    low_freq: float = 50.0,
    high_freq: float = 1000.0,
    harmonic_threshold: float = 0.1,
) -> Tuple[Vector, Vector, Vector, Vector]: ...
def get_dominant_frequencies(
    sig: FloatSequence,
    fs: int = 16_000,
    nfft: int = 512,
    win_len: float = 0.025,
    win_hop: float = 0.010,
    win_type: str = "hamming",
    only_positive: bool = True,
) -> Vector: ...
def cochleagram(
    sig: FloatSequence,
    opts: Optional[CochleagramOptions] = None,
) -> CochleagramOutput: ...
def extract_feats(sig: FloatSequence, fs: int, nfft: int = 512) -> SpectralFeats: ...

def hz2mel(freq: float, conversion: str = "oshaghnessy") -> float: ...
def mel2hz(freq: float, conversion: str = "oshaghnessy") -> float: ...
def hz2bark(freq: float, conversion: str = "wang") -> float: ...
def bark2hz(freq: float, conversion: str = "wang") -> float: ...
def hz2erb(freq: float, conversion: str = "glasberg") -> float: ...
def erb2hz(freq: float, conversion: str = "glasberg") -> float: ...
