#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESP32 Wi-Fi CSI Preprocessing Module for Gesture Recognition
============================================================
Handles:
1. Complex subcarrier reconstruction: z_k = Real + j*Imag
   (Note: in esp-csi, interleaved data array is [Imag_0, Real_0, Imag_1, Real_1, ...])
2. Amplitude and Phase extraction (with phase unwrapping & linear detrending)
3. Null / Guard / Pilot subcarrier removal
4. Spike filtering (Hampel filter / outlier elimination)
5. Temporal smoothing (Savitzky-Golay or Butterworth lowpass filter)
6. Feature Normalization (Z-score or Min-Max scaling)
"""

import json
import numpy as np
from scipy import signal

# IEEE 802.11n 20 MHz (HT20) standard subcarrier structure for 64-point FFT:
# Total subcarriers: 64 (indices 0 to 63 or -32 to 31)
# Null / Guard subcarriers: 0..5, 59..63, and center DC subcarrier (index 32 or index 0 depending on FFT shift)
# Pilot subcarriers in HT20: subcarriers at +/-7 and +/-21
HT20_NULL_GUARD_INDICES = {0, 1, 2, 3, 4, 5, 32, 59, 60, 61, 62, 63}
HT20_PILOT_INDICES = {11, 25, 39, 53} # corresponding to -21, -7, +7, +21 shifted by 32

def parse_csi_data_field(data_field):
    """
    Parses the raw JSON string or list of interleaved integers from esp-csi:
    Format: [Imag_0, Real_0, Imag_1, Real_1, ...]
    Returns complex numpy array of subcarriers.
    """
    if isinstance(data_field, str):
        raw_vals = json.loads(data_field)
    elif isinstance(data_field, (list, tuple, np.ndarray)):
        raw_vals = list(data_field)
    else:
        raise ValueError(f"Unsupported data_field type: {type(data_field)}")

    num_subcarriers = len(raw_vals) // 2
    complex_subcarriers = np.zeros(num_subcarriers, dtype=np.complex64)

    for i in range(num_subcarriers):
        imag = float(raw_vals[i * 2])
        real = float(raw_vals[i * 2 + 1])
        complex_subcarriers[i] = complex(real, imag)

    return complex_subcarriers

def extract_amplitude(complex_csi_matrix):
    """
    Extracts amplitude |z| from complex CSI matrix.
    Input shape: (N_packets, N_subcarriers) or (N_subcarriers,)
    """
    return np.abs(complex_csi_matrix)

def extract_phase(complex_csi_matrix, unwrap=True, detrend=True):
    """
    Extracts phase angle from complex CSI matrix.
    Optionally unwraps along subcarriers and removes Carrier Frequency Offset (CFO) / linear tilt.
    """
    phase = np.angle(complex_csi_matrix)
    if unwrap:
        if phase.ndim == 1:
            phase = np.unwrap(phase)
        else:
            phase = np.unwrap(phase, axis=1)

    if detrend and phase.ndim == 2:
        # Linear regression detrending across subcarrier index to eliminate CFO & SFO
        num_packets, num_subcarriers = phase.shape
        x = np.arange(num_subcarriers)
        for i in range(num_packets):
            # slope and intercept
            slope, intercept = np.polyfit(x, phase[i, :], deg=1)
            phase[i, :] = phase[i, :] - (slope * x + intercept)
    return phase

def remove_null_subcarriers(amplitude_matrix, custom_null_indices=None, auto_detect_zero_variance=False):
    """
    Removes null, guard, and pilot subcarriers.
    amplitude_matrix: shape (N_packets, N_subcarriers)
    Returns: filtered amplitude matrix of active, information-bearing subcarriers.
    """
    matrix = np.array(amplitude_matrix, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)

    num_packets, num_subcarriers = matrix.shape
    to_remove = set()

    if custom_null_indices is not None:
        to_remove.update(custom_null_indices)
    elif num_subcarriers == 64:
        # Standard HT20 64 subcarriers: exactly 16 null/guard/pilot subcarriers
        to_remove.update(HT20_NULL_GUARD_INDICES)
        to_remove.update(HT20_PILOT_INDICES)

    # Optional dynamic auto-detection for non-standard subcarrier counts
    if auto_detect_zero_variance and num_packets > 1:
        stds = np.std(matrix, axis=0)
        means = np.mean(matrix, axis=0)
        dead_subcarriers = np.where((means < 1e-4) | (stds < 1e-5))[0]
        to_remove.update(dead_subcarriers.tolist())

    active_indices = [i for i in range(num_subcarriers) if i not in to_remove]
    active_indices.sort()

    return matrix[:, active_indices], active_indices

def hampel_filter_1d(signal_1d, window_size=7, n_sigmas=3.0):
    """
    Applies Hampel Filter to remove impulse spikes / outliers.
    Replaces outliers with rolling median.
    """
    n = len(signal_1d)
    filtered = np.copy(signal_1d)
    k = window_size // 2

    for i in range(n):
        start = max(0, i - k)
        end = min(n, i + k + 1)
        window = signal_1d[start:end]
        median = np.median(window)
        # Median Absolute Deviation (MAD)
        mad = np.median(np.abs(window - median))
        scale = 1.4826 * mad
        if scale > 1e-9:
            if np.abs(signal_1d[i] - median) > n_sigmas * scale:
                filtered[i] = median
    return filtered

def filter_spikes(amplitude_matrix, window_size=7, n_sigmas=3.0):
    """
    Applies Hampel filter to each subcarrier time-series.
    Input shape: (N_packets, N_subcarriers)
    """
    matrix = np.copy(amplitude_matrix)
    num_packets, num_subcarriers = matrix.shape
    if num_packets < window_size:
        return matrix

    for sc in range(num_subcarriers):
        matrix[:, sc] = hampel_filter_1d(matrix[:, sc], window_size=window_size, n_sigmas=n_sigmas)
    return matrix

def smooth_csi(amplitude_matrix, method="savgol", window_length=11, polyorder=2, cutoff_hz=5.0, fs=100.0):
    """
    Smooths high-frequency noise from CSI amplitudes while preserving gesture motion contours.
    Methods:
      - "savgol": Savitzky-Golay filter (preserves peaks & local dynamics)
      - "butterworth": Butterworth lowpass filter (ideal for human motion 0.5-8Hz)
      - "moving_avg": Simple moving average
    """
    matrix = np.copy(amplitude_matrix)
    num_packets, num_subcarriers = matrix.shape

    if num_packets < 5:
        return matrix

    if method == "savgol":
        # window_length must be odd and <= num_packets
        w_len = window_length if window_length % 2 == 1 else window_length + 1
        if w_len > num_packets:
            w_len = num_packets if num_packets % 2 == 1 else num_packets - 1
        if w_len > polyorder + 1:
            for sc in range(num_subcarriers):
                matrix[:, sc] = signal.savgol_filter(matrix[:, sc], window_length=w_len, polyorder=polyorder)
        return matrix

    elif method == "butterworth":
        # Lowpass filter with cutoff at gesture frequency band (e.g. 5Hz)
        nyquist = 0.5 * fs
        normal_cutoff = min(cutoff_hz / nyquist, 0.99)
        b, a = signal.butter(N=3, Wn=normal_cutoff, btype='low', analog=False)
        for sc in range(num_subcarriers):
            matrix[:, sc] = signal.filtfilt(b, a, matrix[:, sc])
        return matrix

    elif method == "moving_avg":
        kernel = np.ones(window_length) / window_length
        for sc in range(num_subcarriers):
            matrix[:, sc] = np.convolve(matrix[:, sc], kernel, mode='same')
        return matrix

    else:
        raise ValueError(f"Unknown smoothing method: {method}")

def normalize_csi(amplitude_matrix, method="zscore"):
    """
    Normalizes CSI amplitudes across time per subcarrier.
    Methods:
      - "zscore": (x - mean) / std (Standard Normalizer, robust across varying distances)
      - "minmax": scales values to [0, 1] range
    """
    matrix = np.copy(amplitude_matrix)
    if method == "zscore":
        means = np.mean(matrix, axis=0, keepdims=True)
        stds = np.std(matrix, axis=0, keepdims=True)
        stds[stds == 0] = 1.0
        return (matrix - means) / stds

    elif method == "minmax":
        mins = np.min(matrix, axis=0, keepdims=True)
        maxs = np.max(matrix, axis=0, keepdims=True)
        ranges = maxs - mins
        ranges[ranges == 0] = 1.0
        return (matrix - mins) / ranges

    else:
        raise ValueError(f"Unknown normalization method: {method}")

def preprocess_csi_pipeline(raw_csi_complex_or_amp,
                            remove_nulls=True,
                            filter_outliers=True,
                            smooth=True,
                            normalize=True,
                            smooth_method="savgol",
                            norm_method="zscore"):
    """
    Complete end-to-end preprocessing pipeline for gesture recognition.
    Input:
      - raw_csi_complex_or_amp: (N_packets, N_subcarriers) complex or amplitude matrix
    Returns:
      - processed_features: cleaned, smoothed, and normalized matrix ready for gesture classification/ML
      - metadata: dictionary containing active subcarrier indices and intermediate transformations
    """
    # 1. Amplitude Extraction
    if np.iscomplexobj(raw_csi_complex_or_amp):
        amp = extract_amplitude(raw_csi_complex_or_amp)
    else:
        amp = np.array(raw_csi_complex_or_amp, dtype=np.float64)

    metadata = {"initial_subcarriers": amp.shape[1], "packet_count": amp.shape[0]}

    # 2. Remove Null / Guard / Pilot Subcarriers
    if remove_nulls:
        amp, active_indices = remove_null_subcarriers(amp)
        metadata["active_subcarrier_indices"] = active_indices
        metadata["active_subcarriers_count"] = len(active_indices)

    # 3. Spike / Outlier Filtering
    if filter_outliers and amp.shape[0] >= 7:
        amp = filter_spikes(amp, window_size=7, n_sigmas=3.0)

    # 4. Smoothing
    if smooth and amp.shape[0] >= 5:
        amp = smooth_csi(amp, method=smooth_method, window_length=11, polyorder=2)

    # 5. Normalization
    if normalize and amp.shape[0] > 1:
        amp = normalize_csi(amp, method=norm_method)

    return amp, metadata
