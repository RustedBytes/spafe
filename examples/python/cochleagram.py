from __future__ import annotations

import spafe

from _common import ensure_target, plot_heatmap, sine_wave, write_matrix_csv


def main() -> None:
    signal_size = 4096
    sr = 16_000
    signal = sine_wave(frequency=440.0, seconds=signal_size / sr, fs=sr)
    opts = spafe.CochleagramOptions(
        signal_size=signal_size,
        sr=sr,
        env_sr=400,
        filter_n=16,
        low_lim=50.0,
        high_lim=6_000.0,
        sample_factor=2,
        downsampling_window_size=129,
        compression="power",
    )

    output = spafe.cochleagram(signal, opts)

    target = ensure_target()
    write_matrix_csv(target / "cochleagram.csv", output.cochleagram)
    plot_heatmap(
        target / "cochleagram.png",
        output.cochleagram,
        "Cochleagram",
        xlabel="Frame",
        ylabel="Filter",
        colorbar_label="Compressed envelope",
    )

    cols = len(output.cochleagram[0]) if output.cochleagram else 0
    print(f"cochleagram: {len(output.cochleagram)} filters x {cols} frames")
    print(f"first center frequencies: {[round(value, 3) for value in output.center_freqs[:5]]}")
    print(f"wrote outputs to {target}")


if __name__ == "__main__":
    main()
