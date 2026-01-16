#!/usr/bin/env python3
"""
Breath Data Analysis Script

Analyzes accelerometer data to explore inhale/exhale detection algorithms.
Run with: uv run analyze.py [path_to_csv]
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq


def load_data(filepath: str) -> pd.DataFrame:
    """Load breath data from CSV file."""
    df = pd.read_csv(filepath)
    # Convert timestamp from ms to seconds for easier reading
    df['time_sec'] = df['timestamp'] / 1000.0
    return df


def detect_anomalies(df: pd.DataFrame, threshold_std: float = 4.0, max_gap_ms: float = 100) -> tuple[int, int]:
    """
    Detect anomalies at beginning and end of recording (phone movement).

    Uses:
    - Magnitude of acceleration and its rate of change
    - Timestamp gaps (recording interruptions)

    Returns (start_idx, end_idx) - the range of clean data.
    """
    n = len(df)

    # Compute acceleration magnitude
    magnitude = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2).values

    # Compute rate of change (derivative) of magnitude
    mag_diff = np.abs(np.diff(magnitude))

    # Detect timestamp gaps (recording interruptions)
    timestamps = df['timestamp'].values
    ts_diff = np.diff(timestamps)
    ts_gap_anomaly = np.zeros(n, dtype=bool)
    ts_gap_anomaly[:-1] = ts_diff > max_gap_ms  # Mark sample BEFORE the gap

    # Use the middle 60% of data to establish baseline statistics
    baseline_start = int(n * 0.2)
    baseline_end = int(n * 0.8)
    baseline = magnitude[baseline_start:baseline_end]

    mean_mag = baseline.mean()
    std_mag = baseline.std()

    baseline_diff = mag_diff[baseline_start:baseline_end]
    mean_diff = baseline_diff.mean()
    std_diff = baseline_diff.std()

    # Detect magnitude anomalies
    mag_anomaly = np.abs(magnitude - mean_mag) > threshold_std * std_mag
    diff_anomaly = np.zeros(n, dtype=bool)
    diff_anomaly[:-1] = mag_diff > threshold_std * std_diff

    # Combine all anomaly types
    combined_anomaly = mag_anomaly | diff_anomaly | ts_gap_anomaly

    # Find start: skip past any anomalies in first 20%
    check_end_start = int(n * 0.2)
    start_idx = 0

    # Find last anomaly in the first 20% and start after it
    anomaly_in_start = np.where(combined_anomaly[:check_end_start])[0]
    if len(anomaly_in_start) > 0:
        # Start after the last anomaly, plus a small buffer
        start_idx = anomaly_in_start[-1] + 1

    # Find end: first anomaly index in last 20%
    check_start_end = int(n * 0.8)
    end_idx = n
    anomaly_in_end = np.where(combined_anomaly[check_start_end:])[0]
    if len(anomaly_in_end) > 0:
        end_idx = check_start_end + anomaly_in_end[0]

    return start_idx, end_idx


def trim_anomalies(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Automatically trim beginning and end of recording where phone movement anomaly is detected.
    """
    original_len = len(df)
    start_idx, end_idx = detect_anomalies(df)

    trimmed_start = start_idx > 0
    trimmed_end = end_idx < original_len

    if trimmed_start or trimmed_end:
        if verbose:
            if trimmed_start:
                trimmed_duration = start_idx / 60  # Assuming ~60 Hz
                print(f"Trimming first {trimmed_duration:.1f}s ({start_idx} samples) - start anomaly")
            if trimmed_end:
                trimmed_duration = (original_len - end_idx) / 60
                print(f"Trimming last {trimmed_duration:.1f}s ({original_len - end_idx} samples) - end anomaly")

        trimmed_df = df.iloc[start_idx:end_idx].copy()
        # Reset time to start from 0
        trimmed_df['time_sec'] = trimmed_df['time_sec'] - trimmed_df['time_sec'].iloc[0]
        return trimmed_df

    if verbose:
        print("No anomalies detected")
    return df


def compute_pca_projection(df: pd.DataFrame) -> np.ndarray:
    """
    Compute PCA and project data onto principal component.
    This mimics what the web app does during calibration.
    """
    data = df[['x', 'y', 'z']].values

    # Center the data
    mean = data.mean(axis=0)
    centered = data - mean

    # Compute covariance matrix
    cov = np.cov(centered.T)

    # Get eigenvectors (principal components)
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # Sort by eigenvalue (largest first)
    idx = np.argsort(eigenvalues)[::-1]
    principal_component = eigenvectors[:, idx[0]]

    # Project onto principal component
    projected = centered @ principal_component

    return projected.real  # Real part in case of complex numbers


def plot_raw_axes(df: pd.DataFrame, ax: plt.Axes):
    """Plot 1: Raw X, Y, Z accelerometer data over time."""
    ax.plot(df['time_sec'], df['x'], label='X', alpha=0.8, linewidth=0.5)
    ax.plot(df['time_sec'], df['y'], label='Y', alpha=0.8, linewidth=0.5)
    ax.plot(df['time_sec'], df['z'], label='Z', alpha=0.8, linewidth=0.5)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Acceleration (m/s²)')
    ax.set_title('Raw Accelerometer Data (X, Y, Z)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)


def plot_pca_signal(df: pd.DataFrame, pca_signal: np.ndarray, ax: plt.Axes):
    """Plot 2: PCA-projected signal (breathing signal)."""
    ax.plot(df['time_sec'], pca_signal, 'b-', linewidth=0.7)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('PCA Signal')
    ax.set_title('PCA Projection (Primary Motion Axis)')
    ax.grid(True, alpha=0.3)


def plot_smoothed_signal(df: pd.DataFrame, pca_signal: np.ndarray, ax: plt.Axes):
    """Plot 3: Smoothed signal with different window sizes."""
    time = df['time_sec'].values

    # Different smoothing windows
    windows = [5, 15, 30]  # samples (at 60Hz: ~0.08s, 0.25s, 0.5s)

    ax.plot(time, pca_signal, 'gray', alpha=0.3, linewidth=0.5, label='Raw')

    colors = ['blue', 'green', 'red']
    for window, color in zip(windows, colors):
        smoothed = pd.Series(pca_signal).rolling(window=window, center=True).mean()
        ax.plot(time, smoothed, color, linewidth=1,
                label=f'Window={window} (~{window/60:.2f}s)')

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Signal')
    ax.set_title('Smoothed PCA Signal (Moving Average)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)


def plot_derivative(df: pd.DataFrame, pca_signal: np.ndarray, ax: plt.Axes):
    """Plot 4: Signal derivative (rate of change) for detecting transitions."""
    time = df['time_sec'].values

    # Smooth first, then compute derivative
    smoothed = pd.Series(pca_signal).rolling(window=15, center=True).mean().values
    derivative = np.gradient(smoothed, time)

    # Smooth the derivative too
    derivative_smooth = pd.Series(derivative).rolling(window=10, center=True).mean()

    ax.plot(time, derivative_smooth, 'purple', linewidth=0.7)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.fill_between(time, derivative_smooth, 0,
                    where=derivative_smooth > 0, alpha=0.3, color='green', label='Inhale?')
    ax.fill_between(time, derivative_smooth, 0,
                    where=derivative_smooth < 0, alpha=0.3, color='red', label='Exhale?')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Rate of Change')
    ax.set_title('Signal Derivative (Slope) - Inhale/Exhale Detection')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)


def plot_peak_detection(df: pd.DataFrame, pca_signal: np.ndarray, ax: plt.Axes):
    """Plot 5: Peak and trough detection for breath cycle identification."""
    time = df['time_sec'].values

    # Smooth the signal
    smoothed = pd.Series(pca_signal).rolling(window=20, center=True).mean().values

    # Find peaks (inhale maxima) and troughs (exhale minima)
    # distance: minimum samples between peaks (at 60Hz, 120 = 2 seconds min breath cycle)
    peaks, peak_props = signal.find_peaks(smoothed, distance=120, prominence=0.01)
    troughs, trough_props = signal.find_peaks(-smoothed, distance=120, prominence=0.01)

    ax.plot(time, smoothed, 'b-', linewidth=0.7, label='Smoothed Signal')
    ax.scatter(time[peaks], smoothed[peaks], c='green', s=50, marker='^',
               label=f'Peaks (Inhale Max) n={len(peaks)}', zorder=5)
    ax.scatter(time[troughs], smoothed[troughs], c='red', s=50, marker='v',
               label=f'Troughs (Exhale Max) n={len(troughs)}', zorder=5)

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Signal')
    ax.set_title('Peak/Trough Detection - Breath Cycles')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)


def plot_fft_spectrum(df: pd.DataFrame, pca_signal: np.ndarray, ax: plt.Axes):
    """Plot 6: Frequency spectrum to identify breathing rate."""
    # Estimate sample rate from data
    dt = np.diff(df['timestamp'].values).mean() / 1000.0  # Convert to seconds
    sample_rate = 1.0 / dt

    n = len(pca_signal)

    # Compute FFT
    yf = fft(pca_signal - pca_signal.mean())  # Remove DC component
    xf = fftfreq(n, dt)[:n//2]
    power = 2.0/n * np.abs(yf[0:n//2])

    # Only plot up to 1 Hz (breathing is typically 0.1-0.5 Hz)
    mask = xf <= 1.0

    ax.plot(xf[mask], power[mask], 'b-', linewidth=1)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power')
    ax.set_title(f'Frequency Spectrum (Sample Rate: {sample_rate:.1f} Hz)')
    ax.grid(True, alpha=0.3)

    # Find dominant frequency
    peak_idx = np.argmax(power[mask])
    peak_freq = xf[mask][peak_idx]
    if peak_freq > 0:
        breaths_per_min = peak_freq * 60
        ax.axvline(x=peak_freq, color='red', linestyle='--', alpha=0.7)
        ax.annotate(f'Peak: {peak_freq:.3f} Hz\n({breaths_per_min:.1f} breaths/min)',
                    xy=(peak_freq, power[mask][peak_idx]),
                    xytext=(peak_freq + 0.1, power[mask][peak_idx]),
                    fontsize=9)


def plot_adaptive_threshold(df: pd.DataFrame, pca_signal: np.ndarray, ax: plt.Axes):
    """Plot 7: Adaptive threshold using rolling median for zero-crossing detection."""
    time = df['time_sec'].values

    # Smooth the signal
    smoothed = pd.Series(pca_signal).rolling(window=15, center=True).mean()

    # Compute rolling median (adaptive baseline)
    window_sec = 8  # 8-second window like the web app
    window_samples = int(window_sec * 60)  # Assuming 60 Hz
    rolling_median = smoothed.rolling(window=window_samples, center=True, min_periods=1).median()

    # Deviation from median
    deviation = smoothed - rolling_median

    ax.plot(time, deviation, 'b-', linewidth=0.7, label='Deviation from Median')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.fill_between(time, deviation, 0,
                    where=deviation > 0, alpha=0.3, color='green', label='Above Median')
    ax.fill_between(time, deviation, 0,
                    where=deviation < 0, alpha=0.3, color='red', label='Below Median')

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Deviation')
    ax.set_title('Adaptive Threshold (Deviation from Rolling Median)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)


def plot_zero_crossings(df: pd.DataFrame, pca_signal: np.ndarray, ax: plt.Axes):
    """Plot 8: Zero-crossing detection on smoothed deviation."""
    time = df['time_sec'].values

    # Smooth and compute deviation from median
    smoothed = pd.Series(pca_signal).rolling(window=15, center=True).mean().values
    window_samples = int(8 * 60)
    rolling_median = pd.Series(smoothed).rolling(window=window_samples, center=True, min_periods=1).median().values
    deviation = smoothed - rolling_median

    # Find zero crossings
    zero_crossings = np.where(np.diff(np.signbit(deviation)))[0]

    # Classify as inhale start (negative to positive) or exhale start (positive to negative)
    inhale_starts = zero_crossings[deviation[zero_crossings + 1] > 0]
    exhale_starts = zero_crossings[deviation[zero_crossings + 1] < 0]

    ax.plot(time, deviation, 'b-', linewidth=0.7)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    ax.scatter(time[inhale_starts], deviation[inhale_starts], c='green', s=30, marker='|',
               label=f'Inhale Start (n={len(inhale_starts)})', zorder=5)
    ax.scatter(time[exhale_starts], deviation[exhale_starts], c='red', s=30, marker='|',
               label=f'Exhale Start (n={len(exhale_starts)})', zorder=5)

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Deviation')
    ax.set_title('Zero-Crossing Detection (Breath Phase Transitions)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)


def main():
    # Find CSV file
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Auto-find first CSV in records directory
        records_dir = Path(__file__).parent / 'records'
        csv_files = list(records_dir.glob('*.csv'))
        if not csv_files:
            print("No CSV files found in records/ directory")
            print("Usage: uv run analyze.py [path_to_csv]")
            sys.exit(1)
        csv_path = csv_files[0]
        print(f"Using: {csv_path}")

    # Load data
    df = load_data(csv_path)
    print(f"Loaded {len(df)} samples")
    print(f"Duration: {df['time_sec'].max():.1f} seconds")
    print(f"Sample rate: {len(df) / df['time_sec'].max():.1f} Hz")

    # Trim anomalies at beginning and end (phone movement)
    df = trim_anomalies(df)
    print(f"Clean data: {len(df)} samples ({df['time_sec'].max():.1f}s)")

    # Compute PCA projection
    pca_signal = compute_pca_projection(df)

    # Create figure with subplots
    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    fig.suptitle(f'Breath Data Analysis: {Path(csv_path).name}', fontsize=14, fontweight='bold')

    # Generate all plots
    plot_raw_axes(df, axes[0, 0])
    plot_pca_signal(df, pca_signal, axes[0, 1])
    plot_smoothed_signal(df, pca_signal, axes[1, 0])
    plot_derivative(df, pca_signal, axes[1, 1])
    plot_peak_detection(df, pca_signal, axes[2, 0])
    plot_fft_spectrum(df, pca_signal, axes[2, 1])
    plot_adaptive_threshold(df, pca_signal, axes[3, 0])
    plot_zero_crossings(df, pca_signal, axes[3, 1])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
