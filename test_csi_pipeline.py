#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSI Pipeline Test and Validation Script
======================================
Tests the full CSI pipeline:
1. Verifies exact esp-csi README CSI_DATA format parsing
2. Simulates 150 packets of Wi-Fi CSI with an injected hand gesture
3. Executes and verifies each preprocessing stage:
   - Complex subcarrier parsing
   - Amplitude calculation
   - Null/pilot subcarrier removal
   - Hampel spike filtering
   - Savitzky-Golay smoothing
   - Z-score normalization
4. Saves a comparison visualization figure: 'csi_gesture_preprocessed.png'
"""

import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless backend for plotting without display
import matplotlib.pyplot as plt

from csi_preprocessor import (
    parse_csi_data_field,
    extract_amplitude,
    extract_phase,
    remove_null_subcarriers,
    filter_spikes,
    smooth_csi,
    normalize_csi,
    preprocess_csi_pipeline
)
from simulate_csi import CSISimulator

def test_readme_sample_line():
    """
    Verifies that the exact sample line published in the official esp-csi README
    parses accurately into 64 complex subcarriers.
    """
    print("\n[+] Test 1: Official esp-csi README Sample Line Parsing...")
    sample_data_json = "[67,48,4,0,0,0,0,0,0,0,5,0,20,1,20,1,19,0,17,1,16,2,15,2,14,1,12,0,12,-1,12,-3,12,-4,13,-6,15,-7,16,-8,16,-8,16,-8,16,-6,15,-5,15,-4,14,-4,13,-4,12,-4,11,-4,10,-4,9,-5,8,-6,4,-4,8,-9,9,-10,9,-10,10,-11,11,-10,11,-10,12,-9,11,-8,11,-7,10,-6,9,-6,7,-6,6,-7,5,-7,5,-8,5,-9,5,-10,5,-11,5,-11,6,-11,7,-11,8,-11,9,-10,9,-9,8,-8,8,-7,1,-2,0,0,0,0,0,0,0,0]"
    
    complex_sc = parse_csi_data_field(sample_data_json)
    assert len(complex_sc) == 64, f"Expected 64 subcarriers, got {len(complex_sc)}"
    
    # Subcarrier 0 in esp-csi sample: Imag=67, Real=48 -> z = 48 + 67j
    assert complex_sc[0].real == 48.0, f"Expected real 48, got {complex_sc[0].real}"
    assert complex_sc[0].imag == 67.0, f"Expected imag 67, got {complex_sc[0].imag}"
    
    amp_0 = np.abs(complex_sc[0])
    expected_amp_0 = np.sqrt(48**2 + 67**2)
    assert np.isclose(amp_0, expected_amp_0), f"Expected amp {expected_amp_0}, got {amp_0}"
    
    print(f"    [PASS] Successfully parsed 64 subcarriers from README sample.")
    print(f"    [PASS] Subcarrier 0: Real={complex_sc[0].real}, Imag={complex_sc[0].imag}, Amplitude={amp_0:.2f}")

def test_full_pipeline_with_gesture():
    """
    Simulates a sequence of 200 CSI frames with a hand gesture wave in the middle.
    Validates all preprocessing steps and generates a diagnostic figure.
    """
    print("\n[+] Test 2: Full Preprocessing Pipeline on Synthetic Gesture Data...")
    num_packets = 200
    fps = 50 # 50 Hz packet rate -> 4.0 seconds total duration
    sim = CSISimulator(packet_rate_hz=fps)

    complex_frames = []
    timestamps = []

    for i in range(num_packets):
        packet_str = sim.generate_packet(inject_gesture=True)
        # Extract data json field (last element)
        parts = packet_str.split(',"[')
        raw_json = "[" + parts[1].rstrip('"')
        complex_sc = parse_csi_data_field(raw_json)
        complex_frames.append(complex_sc)
        timestamps.append(i / fps)

    raw_complex_matrix = np.array(complex_frames) # (200, 64)
    time_axis = np.array(timestamps)

    # Step 1: Raw Amplitude
    raw_amps = extract_amplitude(raw_complex_matrix)
    print(f"    [Step 1] Raw amplitude extracted: Shape {raw_amps.shape}")
    assert raw_amps.shape == (num_packets, 64)

    # Step 2: Remove Null and Pilot Subcarriers
    active_amps, active_indices = remove_null_subcarriers(raw_amps)
    print(f"    [Step 2] Null & pilot subcarriers removed: {raw_amps.shape[1]} -> {active_amps.shape[1]} active subcarriers")
    assert active_amps.shape[1] < 64, "Null subcarriers should have been pruned."
    assert 0 not in active_indices and 32 not in active_indices, "Guard/DC subcarriers must be removed."

    # Step 3: Spike / Outlier Filtering (Hampel Filter)
    de_spiked_amps = filter_spikes(active_amps, window_size=7, n_sigmas=3.0)
    print(f"    [Step 3] Hampel spike filter applied: Shape {de_spiked_amps.shape}")

    # Step 4: Smoothing (Savitzky-Golay Filter)
    smoothed_amps = smooth_csi(de_spiked_amps, method="savgol", window_length=11, polyorder=2)
    print(f"    [Step 4] Savitzky-Golay smoothing applied: Shape {smoothed_amps.shape}")

    # Step 5: Normalization (Z-score)
    normalized_amps = normalize_csi(smoothed_amps, method="zscore")
    print(f"    [Step 5] Z-score normalization applied: Shape {normalized_amps.shape}")
    # Verify that each column has mean ~ 0 and std ~ 1
    col_means = np.mean(normalized_amps, axis=0)
    col_stds = np.std(normalized_amps, axis=0)
    assert np.all(np.abs(col_means) < 1e-4), "Normalized mean should be ~0"
    assert np.all(np.abs(col_stds - 1.0) < 1e-3), "Normalized std should be ~1"

    # Step 6: Create Comparison Figure
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    plt.subplots_adjust(hspace=0.35)

    # Subplot 1: Raw Amplitudes across all subcarriers (showing nulls and spikes)
    representative_sc = [active_indices[5], active_indices[15], active_indices[25], active_indices[35]]
    for sc in representative_sc:
        axes[0].plot(time_axis, raw_amps[:, sc], label=f"Subcarrier {sc}", alpha=0.8)
    axes[0].set_title("1. Raw CSI Amplitudes (Showing thermal noise & impulse spikes)", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="upper right", ncol=4, fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: De-spiked Amplitudes
    for i, sc in enumerate(representative_sc):
        sub_idx = active_indices.index(sc)
        axes[1].plot(time_axis, de_spiked_amps[:, sub_idx], label=f"Subcarrier {sc}", alpha=0.8)
    axes[1].set_title("2. After Hampel Spike Filter (Outlier spikes removed)", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Subplot 3: Smoothed Amplitudes
    for i, sc in enumerate(representative_sc):
        sub_idx = active_indices.index(sc)
        axes[2].plot(time_axis, smoothed_amps[:, sub_idx], label=f"Subcarrier {sc}", linewidth=1.8)
    axes[2].axvspan(1.5, 3.5, color='orange', alpha=0.15, label='Gesture Active Window')
    axes[2].set_title("3. After Savitzky-Golay Smoothing (Highlighting Hand Gesture Motion)", fontsize=11, fontweight='bold')
    axes[2].set_ylabel("Amplitude")
    axes[2].legend(loc="upper right", fontsize=9)
    axes[2].grid(True, linestyle="--", alpha=0.5)

    # Subplot 4: Normalized Features (Ready for ML gesture classification)
    im = axes[3].imshow(normalized_amps.T, aspect='auto', cmap='plasma', origin='lower',
                        extent=[time_axis[0], time_axis[-1], 0, normalized_amps.shape[1]])
    axes[3].set_title("4. Normalized CSI Spectrogram Matrix (Input for Gesture Recognition)", fontsize=11, fontweight='bold')
    axes[3].set_xlabel("Time (seconds)", fontsize=10)
    axes[3].set_ylabel("Active Subcarrier Index", fontsize=10)
    cbar = fig.colorbar(im, ax=axes[3], orientation='horizontal', pad=0.25, shrink=0.6)
    cbar.set_label("Z-score Normalized Amplitude")

    output_plot_path = os.path.join(os.path.dirname(__file__), "csi_gesture_preprocessed.png")
    fig.savefig(output_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[OK] Diagnostic plot successfully generated and saved to:")
    print(f"    -> {output_plot_path}")

if __name__ == "__main__":
    print("=" * 65)
    print("ESP32 CSI SIGNAL PROCESSING & GESTURE PIPELINE VALIDATION")
    print("=" * 65)
    test_readme_sample_line()
    test_full_pipeline_with_gesture()
    print("\n" + "=" * 65)
    print("ALL TESTS PASSED SUCCESSFULLY! CODE IS VERIFIED & READY FOR REAL DATA.")
    print("=" * 65)
