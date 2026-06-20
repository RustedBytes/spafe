from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "target" / "python-examples"


def sine_wave(
    frequency: float = 440.0,
    seconds: float = 1.0,
    fs: int = 16_000,
) -> list[float]:
    samples = int(seconds * fs)
    return [
        math.sin(2.0 * math.pi * frequency * idx / fs)
        for idx in range(samples)
    ]


def ensure_target() -> Path:
    TARGET.mkdir(parents=True, exist_ok=True)
    return TARGET


def write_matrix_csv(path: Path, matrix: list[list[float]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(matrix)


def write_vector_csv(path: Path, header: list[str], rows: list[list[float]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def plot_heatmap(
    path: Path,
    matrix: list[list[float]],
    title: str,
    xlabel: str,
    ylabel: str,
    colorbar_label: str = "Value",
) -> None:
    if not matrix or not matrix[0]:
        return

    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_filter_bank(
    path: Path,
    rows: list[list[float]],
    x_values: list[float],
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    for row in rows:
        ax.plot(x_values, row, linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Weight")
    ax.grid(True, alpha=0.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_pitch_tracks(
    path: Path,
    times: list[float],
    pitches: list[float],
    dominant: list[float],
    win_hop: float,
) -> None:
    dominant_times = [idx * win_hop for idx in range(len(dominant))]
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    ax.plot(times, pitches, label="YIN pitch", linewidth=1.5)
    ax.plot(dominant_times, dominant, label="Dominant frequency", linewidth=1.0, alpha=0.8)
    ax.set_title("Pitch and Dominant Frequency")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)
