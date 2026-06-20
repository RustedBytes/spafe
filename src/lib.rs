//! Native Rust implementation of the core `spafe` audio feature extraction APIs.
//!
//! The crate keeps the Python package's module vocabulary while exposing idiomatic
//! Rust option structs and `Result`-based errors.
//!
//! Enable the `portable-simd` Cargo feature on nightly Rust to use `std::simd`
//! in the crate's hot numeric kernels.

#![cfg_attr(feature = "portable-simd", feature(portable_simd))]

mod core;
mod simd;

pub use core::{
    BarkConversionApproach, CqccOptions, ErbConversionApproach, FeatureOptions, FilterBankOptions,
    Matrix, MelConversionApproach, Normalization, Result, Scale, SlidingWindow, SpafeError,
    SpectrogramOutput, Vector, WindowType,
};

pub(crate) use core::{
    apply_post_processing, cepstral_from_spectrogram, checked_high_freq, dct_rows, linspace,
    replace_zero, spectrogram_with_fbanks, weighted_power_projection,
};

/// Cochleagram generation with ERB half-cosine filters, envelopes, downsampling, and compression.
pub mod cochleagram;
/// Filter-bank builders for Mel, inverse Mel, linear, Bark, and gammatone scales.
pub mod fbanks;
/// Spectrogram and cepstral feature extractors.
pub mod features;
/// Fundamental and dominant frequency estimation helpers.
pub mod frequencies;
/// Convenient glob-style re-exports for common crate APIs.
pub mod prelude;
/// Spectral descriptor helpers and aggregate descriptor output.
pub mod spfeats;
/// Converters, preprocessing, cepstral, spectral, exception, and visualization utilities.
pub mod utils;

#[cfg(test)]
mod tests;
