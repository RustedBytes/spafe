from __future__ import annotations

import spafe

from _common import ensure_target, plot_heatmap, sine_wave, write_matrix_csv


def main() -> None:
    fs = 16_000
    signal = sine_wave(frequency=440.0, seconds=1.0, fs=fs)
    opts = spafe.FeatureOptions(fs=fs, nfft=256, nfilts=24, win_hop=0.02)

    mfcc = spafe.mfcc(signal, opts)
    gfcc = spafe.gfcc(signal, opts)
    spectrogram = spafe.mel_spectrogram(signal, opts)

    target = ensure_target()
    write_matrix_csv(target / "mfcc.csv", mfcc)
    write_matrix_csv(target / "gfcc.csv", gfcc)
    plot_heatmap(
        target / "mfcc.png",
        mfcc,
        "MFCC",
        xlabel="Coefficient",
        ylabel="Frame",
    )
    plot_heatmap(
        target / "mel_spectrogram.png",
        spectrogram.features,
        "Mel Spectrogram",
        xlabel="Filter",
        ylabel="Frame",
        colorbar_label="Energy",
    )

    print(f"mfcc: {len(mfcc)} frames x {len(mfcc[0]) if mfcc else 0} coefficients")
    print(f"gfcc: {len(gfcc)} frames x {len(gfcc[0]) if gfcc else 0} coefficients")
    print(f"wrote outputs to {target}")


if __name__ == "__main__":
    main()
