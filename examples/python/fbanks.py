from __future__ import annotations

import spafe

from _common import ensure_target, plot_filter_bank, write_matrix_csv


def main() -> None:
    opts = spafe.FilterBankOptions(
        nfilts=24,
        nfft=512,
        fs=16_000,
        high_freq=8_000.0,
    )
    fbanks, centers = spafe.linear_filter_banks(opts)
    freqs = [idx * opts.fs / opts.nfft for idx in range(opts.nfft // 2 + 1)]

    target = ensure_target()
    write_matrix_csv(target / "linear_fbanks.csv", fbanks)
    plot_filter_bank(target / "linear_fbanks.png", fbanks, freqs, "Linear Filter Bank")

    print(f"filter bank: {len(fbanks)} filters x {len(fbanks[0]) if fbanks else 0} bins")
    print(f"first centers: {[round(value, 3) for value in centers[:5]]}")
    print(f"wrote outputs to {target}")


if __name__ == "__main__":
    main()
