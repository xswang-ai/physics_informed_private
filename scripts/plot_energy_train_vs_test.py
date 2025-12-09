#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, FFMpegWriter  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare kinetic energy evolution between training and test NS2D datasets."
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="/scratch3/wan410/operator_learning_data/pino_ns2d/NS_fft_Re500_T4000.npy",
        help="Path to training vorticity dataset (.npy).",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="/scratch3/wan410/operator_learning_data/pino_ns2d/NS_Re500_s256_T100_test.npy",
        help="Path to testing vorticity dataset (.npy).",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Sample index to visualize from each dataset.",
    )
    parser.add_argument(
        "--train_dt",
        type=float,
        default=1.0,
        help="Time step for the training data (seconds).",
    )
    parser.add_argument(
        "--test_dt",
        type=float,
        default=0.5,
        help="Native time step for the test data before temporal downsampling.",
    )
    parser.add_argument(
        "--test_time_downsample",
        type=int,
        default=2,
        help="Temporal stride applied to the test data.",
    )
    parser.add_argument(
        "--test_space_downsample",
        type=int,
        default=4,
        help="Spatial stride applied to the test data.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="saved_plots/energy_train_vs_test.mp4",
        help="Output mp4 path for the energy animation.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI for the saved animation.",
    )
    return parser.parse_args()


def load_sample(path: str, sample_idx: int) -> np.ndarray:
    data = np.load(path, mmap_mode="r")
    if data.ndim != 4:
        raise ValueError(f"Expected a 4D array in {path}, got shape {data.shape}.")
    if not (0 <= sample_idx < data.shape[0]):
        raise IndexError(f"sample_idx={sample_idx} out of range for shape {data.shape}.")
    return np.array(data[sample_idx])


def time_first(sample: np.ndarray) -> np.ndarray:
    if sample.ndim != 3:
        raise ValueError(f"Sample must be 3D (time, x, y or x, y, time), got {sample.shape}.")
    shape = sample.shape
    size_counts = {dim: shape.count(dim) for dim in set(shape)}
    time_axis_candidates = [i for i, dim in enumerate(shape) if size_counts.get(dim, 0) == 1]
    time_axis = time_axis_candidates[0] if time_axis_candidates else 0
    return np.moveaxis(sample, time_axis, 0)


def downsample_test(series: np.ndarray, t_stride: int, s_stride: int) -> np.ndarray:
    return series[::t_stride, ::s_stride, ::s_stride]


def velocity_from_vorticity_np(w_slice: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nx, ny = w_slice.shape
    kx = np.fft.fftfreq(nx)
    ky = np.fft.fftfreq(ny)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    lap = kx_grid**2 + ky_grid**2
    lap[0, 0] = 1.0
    w_hat = np.fft.fft2(w_slice)
    psi_hat = w_hat / lap
    ux_hat = 1j * ky_grid * psi_hat
    uy_hat = -1j * kx_grid * psi_hat
    ux = np.fft.ifft2(ux_hat).real
    uy = np.fft.ifft2(uy_hat).real
    return ux, uy


def kinetic_energy_series(vorticity_series: np.ndarray, Lx: float = 2 * np.pi, Ly: float = 2 * np.pi):
    energies = []
    area = Lx * Ly
    for frame in vorticity_series:
        ux, uy = velocity_from_vorticity_np(frame)
        energies.append(0.5 * area * np.mean(ux**2 + uy**2))
    return np.asarray(energies)


def build_animation(train_t, train_e, test_t, test_e, output_path: str, dpi: int):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_xlabel("Time")
    ax.set_ylabel("Kinetic energy")
    ax.set_title("Energy evolution: train vs test")

    min_e = min(train_e.min(), test_e.min())
    max_e = max(train_e.max(), test_e.max())
    e_pad = 0.05 * (max_e - min_e + 1e-9)
    ax.set_xlim(0.0, max(train_t[-1], test_t[-1]))
    ax.set_ylim(min_e - e_pad, max_e + e_pad)

    line_train, = ax.plot([], [], label="Train", color="tab:blue")
    line_test, = ax.plot([], [], label="Test", color="tab:orange")
    marker_train, = ax.plot([], [], "o", color="tab:blue", markersize=4)
    marker_test, = ax.plot([], [], "o", color="tab:orange", markersize=4)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    frames = max(len(train_t), len(test_t))
    print("train_t: ", train_t, "train_e: ", train_e, "test_t: ", test_t, "test_e: ", test_e)
    def init():
        for artist in (line_train, line_test, marker_train, marker_test):
            artist.set_data([], [])
        return line_train, line_test, marker_train, marker_test

    def update(i):
        idx_train = min(i, len(train_t) - 1)
        idx_test = min(i, len(test_t) - 1)
        line_train.set_data(train_t[: idx_train + 1], train_e[: idx_train + 1])
        line_test.set_data(test_t[: idx_test + 1], test_e[: idx_test + 1])
        marker_train.set_data(train_t[idx_train], train_e[idx_train])
        marker_test.set_data(test_t[idx_test], test_e[idx_test])
        return line_train, line_test, marker_train, marker_test

    anim = FuncAnimation(fig, update, frames=frames, init_func=init, interval=80, blit=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = FFMpegWriter(fps=12, bitrate=1800)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()

    train_sample = time_first(load_sample(args.train_path, args.sample_idx))
    test_sample = time_first(load_sample(args.test_path, args.sample_idx))
    test_sample = downsample_test(test_sample, args.test_time_downsample, args.test_space_downsample)

    train_energy = kinetic_energy_series(train_sample)
    test_energy = kinetic_energy_series(test_sample)

    train_times = np.arange(train_sample.shape[0]) * args.train_dt
    downsampled_dt_test = args.test_dt * args.test_time_downsample
    test_times = np.arange(test_sample.shape[0]) * downsampled_dt_test

    build_animation(train_times, train_energy, test_times, test_energy, args.output, args.dpi)
    print(f"Saved energy animation to {args.output}")


if __name__ == "__main__":
    main()
