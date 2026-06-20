from __future__ import annotations

import spafe

from _common import ensure_target, plot_pitch_tracks, sine_wave, write_vector_csv


def main() -> None:
    fs = 16_000
    signal = sine_wave(frequency=440.0, seconds=1.0, fs=fs)
    pitches, harmonic_rates, argmins, times = spafe.compute_yin(
        signal,
        fs=fs,
        win_len=0.03,
        win_hop=0.015,
        low_freq=50.0,
        high_freq=1_000.0,
        harmonic_threshold=0.1,
    )
    dominant = spafe.get_dominant_frequencies(
        signal,
        fs=fs,
        nfft=512,
        win_len=0.025,
        win_hop=0.010,
    )

    target = ensure_target()
    write_vector_csv(
        target / "yin.csv",
        ["time", "pitch", "harmonic_rate", "argmin"],
        [[time, pitch, rate, argmin] for time, pitch, rate, argmin in zip(times, pitches, harmonic_rates, argmins)],
    )
    write_vector_csv(
        target / "dominant_frequencies.csv",
        ["frame", "frequency"],
        [[float(idx), value] for idx, value in enumerate(dominant)],
    )
    plot_pitch_tracks(
        target / "pitch_tracks.png",
        times,
        pitches,
        dominant,
        win_hop=0.010,
    )

    voiced = [pitch for pitch in pitches if pitch > 0.0]
    avg_pitch = sum(voiced) / len(voiced) if voiced else 0.0
    print(f"yin frames: {len(pitches)}, average voiced pitch: {avg_pitch:.3f} Hz")
    print(f"dominant frequency frames: {len(dominant)}")
    print(f"wrote outputs to {target}")


if __name__ == "__main__":
    main()
