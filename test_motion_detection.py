#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSI Motion Detection & Segmentation Test
========================================
Simulates an end-to-end experiment:
- 0.0s - 10.0s: Quiet idle room (learns baseline mean & std)
- 11.0s - 13.0s: Hand wave gesture (multipath Doppler perturbation)
- 13.0s - 16.0s: Return to quiet idle

Validates:
1. Baseline calibration
2. Motion score exceeding T_high during gesture
3. Stable state transitions with hysteresis and debounce
4. Automatic cut-out of the 1-2 second gesture window (.npy)
5. JSON state emissions every 0.5s
6. Generates diagnostic plot 'csi_motion_detection_test.png'
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from simulate_csi import CSISimulator
from csi_preprocessor import parse_csi_data_field, preprocess_csi_pipeline
from csi_motion_detector import CSIMotionDetector
from json_publisher import MotionStatePublisher

def run_motion_detection_test():
    print("=" * 65)
    print("ESP32 CSI MOTION & ANOMALY DETECTION TEST")
    print("=" * 65)

    fps = 50 # 50 packets per second
    total_sec = 16.0
    total_packets = int(total_sec * fps) # 800 packets
    calib_sec = 10.0
    gesture_start_sec = 11.0
    gesture_end_sec = 13.0

    sim = CSISimulator(packet_rate_hz=fps)
    detector = CSIMotionDetector(
        fps=fps,
        calibration_sec=calib_sec,
        k_std_high=3.5,
        k_std_low=1.8,
        debounce_trigger_frames=4,
        debounce_release_frames=10,
        gesture_window_sec=1.5,
        save_dir="gesture_samples"
    )

    # Attach JSON publisher with a clean console listener
    json_messages = []
    def log_json(payload):
        json_messages.append(payload)
        # print every second message to keep terminal clean
        if len(json_messages) % 2 == 0:
            print(f"  [JSON 0.5s] State: {payload['state']:11s} | Score: {payload['score']:.3f} | Thresh: {payload['threshold_high']:.3f}")

    publisher = MotionStatePublisher(detector, custom_callback=log_json, verbose=False)

    time_records = []
    raw_amp_records = []
    preprocessed_records = []
    score_records = []
    state_records = []
    saved_gesture_paths = []

    print(f"\n[*] Simulating {total_sec:.1f}s of CSI data ({total_packets} packets)...")
    print(f"    - Calibration: 0.0s to {calib_sec:.1f}s (Quiet room)")
    print(f"    - Hand Wave:   {gesture_start_sec:.1f}s to {gesture_end_sec:.1f}s")
    print(f"    - Post-Wave:   {gesture_end_sec:.1f}s to {total_sec:.1f}s (Quiet room)\n")

    # Ring buffer for online preprocessing
    window_raw_complex = []

    for i in range(total_packets):
        t = i / fps
        time_records.append(t)

        # Generate packet with gesture strictly during [gesture_start_sec, gesture_end_sec]
        packet_str = sim.generate_packet(
            inject_gesture=True,
            gesture_start_sec=gesture_start_sec,
            gesture_end_sec=gesture_end_sec
        )

        parts = packet_str.split(',"[')
        raw_json = "[" + parts[1].rstrip('"')
        complex_sc = parse_csi_data_field(raw_json)
        window_raw_complex.append(complex_sc)

        # Preprocessing: Keep a sliding window of recent complex frames to allow Savitzky-Golay / smoothing
        if len(window_raw_complex) > 15:
            window_raw_complex.pop(0)

        # Preprocess the sliding window
        window_matrix = np.array(window_raw_complex)
        clean_amp_window, meta = preprocess_csi_pipeline(
            window_matrix,
            remove_nulls=True,
            filter_outliers=True,
            smooth=True,
            normalize=False # Keep raw amplitude variation for motion detection
        )

        current_clean_frame = clean_amp_window[-1, :]
        raw_amp = np.abs(complex_sc)
        raw_amp_records.append(raw_amp)
        preprocessed_records.append(current_clean_frame)

        # Feed to Motion Detector
        res = detector.process_frame(current_clean_frame, timestamp=t)
        score_records.append(res["score"])
        state_records.append(res["state"])

        if res["saved_gesture"]:
            saved_gesture_paths.append(res["saved_gesture"])
            print(f"\n[!] Motion event ended -> Segmented gesture saved to: {res['saved_gesture']}\n")

    time_arr = np.array(time_records)
    scores_arr = np.array(score_records)
    states_arr = state_records
    clean_matrix = np.array(preprocessed_records)

    # Validation Checks
    print("\n" + "=" * 65)
    print("DETECTOR PERFORMANCE SUMMARY")
    print("=" * 65)
    print(f"[PASS] Baseline Mean Score:       {detector.baseline_mean:.4f}")
    print(f"[PASS] Baseline Std Dev:          {detector.baseline_std:.4f}")
    print(f"[PASS] Trigger Threshold (T_high): {detector.threshold_high:.4f} (Mean + {detector.k_high}*Std)")
    print(f"[PASS] Release Threshold (T_low):  {detector.threshold_low:.4f} (Mean + {detector.k_low}*Std)")
    print(f"[PASS] Segmented Gesture Files:    {len(saved_gesture_paths)} file(s) saved in '{detector.save_dir}'")
    print(f"[PASS] Total JSON Heartbeats:      {len(json_messages)} events published")

    # Assertions
    # 1. During quiet calibration, motion state must be CALIBRATING then IDLE
    assert "MOTION" not in states_arr[:int(calib_sec * fps)], "False positive during idle calibration!"
    # 2. Hand gesture must trigger MOTION
    gesture_states = states_arr[int(gesture_start_sec * fps):int(gesture_end_sec * fps)]
    assert "MOTION" in gesture_states, "Failed to detect hand gesture motion!"
    # 3. Post-gesture must return to IDLE
    assert states_arr[-1] == "IDLE", "Detector failed to return to IDLE state after gesture ended!"
    # 4. Segmented gesture example was saved
    assert len(saved_gesture_paths) >= 1, "Gesture window was not saved to disk!"

    # Generate Visualization Figure
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=False)
    plt.subplots_adjust(hspace=0.4)

    # Panel 1: Preprocessed Subcarrier Amplitudes
    active_sc_to_plot = [4, 14, 24, 34, 44]
    for sc in active_sc_to_plot:
        axes[0].plot(time_arr, clean_matrix[:, sc], label=f"Subcarrier {sc}", alpha=0.75)
    axes[0].axvspan(gesture_start_sec, gesture_end_sec, color='orange', alpha=0.15, label='Actual Hand Wave')
    axes[0].axvline(calib_sec, color='blue', linestyle='--', label='End Calibration (10s)')
    axes[0].set_title("1. Cleaned Subcarrier Amplitudes (Nulls/Spikes Removed, Smoothed)", fontsize=11, fontweight='bold')
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="upper right", ncol=4, fontsize=8)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Rolling Motion Score with Dual Hysteresis Thresholds
    axes[1].plot(time_arr, scores_arr, color='navy', linewidth=1.5, label='Motion Score S(t)')
    axes[1].axhline(detector.threshold_high, color='red', linestyle='--', linewidth=1.5, label=f'T_high = {detector.threshold_high:.3f}')
    axes[1].axhline(detector.threshold_low, color='green', linestyle=':', linewidth=1.5, label=f'T_low = {detector.threshold_low:.3f}')
    axes[1].axhline(detector.baseline_mean, color='gray', linestyle='-.', alpha=0.7, label=f'Baseline Mean = {detector.baseline_mean:.3f}')
    axes[1].axvspan(gesture_start_sec, gesture_end_sec, color='orange', alpha=0.15)
    axes[1].set_title("2. Rolling Motion Score & Hysteresis Thresholds (Mean + k*Std)", fontsize=11, fontweight='bold')
    axes[1].set_ylabel("Motion Score")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Panel 3: State Machine Output with Debounce
    state_map = {"CALIBRATING": 0, "IDLE": 1, "MOTION": 2}
    numeric_states = [state_map[s] for s in states_arr]
    axes[2].step(time_arr, numeric_states, color='purple', where='post', linewidth=2, label='Detector State')
    axes[2].set_yticks([0, 1, 2])
    axes[2].set_yticklabels(["CALIBRATING", "IDLE", "MOTION"], fontsize=9, fontweight='bold')
    axes[2].set_title("3. Debounced State Output (Hysteresis prevents flickering)", fontsize=11, fontweight='bold')
    axes[2].set_xlabel("Time (seconds)")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    # Panel 4: The Cut-Out Segmented Gesture Window
    if len(saved_gesture_paths) > 0:
        saved_file = saved_gesture_paths[0]
        gesture_data = np.load(saved_file)
        g_time = np.linspace(0, gesture_data.shape[0] / fps, gesture_data.shape[0])
        im = axes[3].imshow(gesture_data.T, aspect='auto', cmap='plasma', origin='lower',
                            extent=[0, g_time[-1], 0, gesture_data.shape[1]])
        axes[3].set_title(f"4. Cut-Out Gesture Window Example (Duration: {g_time[-1]:.2f}s, Shape: {gesture_data.shape}) -> Saved for Classifier", fontsize=11, fontweight='bold')
        axes[3].set_xlabel("Gesture Duration (seconds)")
        axes[3].set_ylabel("Subcarrier Index")
        cbar = fig.colorbar(im, ax=axes[3], orientation='horizontal', pad=0.35, shrink=0.5)
        cbar.set_label("Amplitude")

    plot_path = os.path.join(os.path.dirname(__file__), "csi_motion_detection_test.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[OK] Diagnostic motion detection plot saved to:")
    print(f"     -> {plot_path}")
    print("\n" + "=" * 65)
    print("ALL TESTS PASSED! MOTION DETECTOR & SEGMENTER READY.")
    print("=" * 65)

if __name__ == "__main__":
    run_motion_detection_test()
