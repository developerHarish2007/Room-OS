#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
90-Second Fast CSI Polling & Hand-Wave Analysis Experiment
=========================================================
Polls csi_data.json as fast as possible for 90 seconds.
- Filters out duplicate raw_iq frames (counting skipped duplicates).
- Computes mean amplitude across active subcarriers (using process_raw_iq logic).
- Segments into three 30s blocks:
    Block 1 (0-30s):  Idle 1
    Block 2 (30-60s): Hand Wave
    Block 3 (60-90s): Idle 2
- Computes mean & standard deviation for each block.
- Generates publication-ready diagnostic plot 'csi_90s_experiment.png'.
"""

import sys
import time
import json
import math
import urllib.request
import urllib.error
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

URL = "https://espcsi-default-rtdb.asia-southeast1.firebasedatabase.app/csi_data.json"
TOTAL_DURATION = 90.0

KNOWN_NULL_OR_META_INDICES = {
    0, 1,               # Header / RSSI metadata
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, # Central null band
    64,                 # Center transition marker
    93, 94, 95, 96, 97, 98, 99                  # High-band null/guard
}

def compute_mean_active_amplitude(raw_iq):
    """Computes mean amplitude across all non-null subcarriers."""
    total_pairs = len(raw_iq) // 2
    active_amps = []

    for sc_idx in range(total_pairs):
        q = float(raw_iq[sc_idx * 2])
        i = float(raw_iq[sc_idx * 2 + 1])
        amp = math.sqrt(i**2 + q**2)

        is_null = (
            sc_idx in KNOWN_NULL_OR_META_INDICES or
            (abs(i) < 1e-4 and abs(q) < 1e-4) or
            (i == -1.0 and q == -1.0)
        )
        if not is_null:
            active_amps.append(amp)

    if not active_amps:
        return 0.0
    return float(np.mean(active_amps))

def run_experiment():
    print("=" * 75)
    print("STARTING 90-SECOND FAST CSI POLLING EXPERIMENT")
    print("=" * 75)
    print("Protocol:")
    print("  - [00s - 30s]: IDLE 1 (Hands away)")
    print("  - [30s - 60s]: WAVING (Wave hand over boards)")
    print("  - [60s - 90s]: IDLE 2 (Hands away again)")
    print("=" * 75)

    readings = []
    skipped_count = 0
    poll_count = 0
    prev_raw_iq = None

    req = urllib.request.Request(URL, headers={"User-Agent": "CSI-FastPoller/1.0"})

    start_time = time.time()
    last_block_announced = -1

    while True:
        now = time.time()
        elapsed = now - start_time
        if elapsed >= TOTAL_DURATION:
            break

        # Announce block transitions
        current_block = int(elapsed // 30)
        if current_block != last_block_announced:
            if current_block == 0:
                print("\n>>> [BLOCK 1: 0-30s] IDLE 1 STARTED (Keep hands away) <<<", flush=True)
            elif current_block == 1:
                print("\n>>> [BLOCK 2: 30-60s] WAVING STARTED (Wave your hand now!) <<<", flush=True)
            elif current_block == 2:
                print("\n>>> [BLOCK 3: 60-90s] IDLE 2 STARTED (Stop waving, keep hands away) <<<", flush=True)
            last_block_announced = current_block

        poll_count += 1
        try:
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                raw_iq = data.get("raw_iq", [])
        except Exception as e:
            time.sleep(0.05)
            continue

        if not raw_iq or len(raw_iq) < 10:
            time.sleep(0.05)
            continue

        # Check deduplication
        if prev_raw_iq is not None and raw_iq == prev_raw_iq:
            skipped_count += 1
            # Brief pause before retrying to allow upstream RTDB to update
            time.sleep(0.04)
            continue

        prev_raw_iq = raw_iq
        mean_amp = compute_mean_active_amplitude(raw_iq)

        readings.append({
            "rel_time": elapsed,
            "epoch_time": now,
            "mean_amplitude": mean_amp
        })

        # Brief sleep between requests
        time.sleep(0.04)

    total_time = time.time() - start_time
    print(f"\n[OK] Polling finished in {total_time:.2f} seconds.")
    print(f"     Total HTTP polls made: {poll_count}")
    print(f"     Unique readings captured: {len(readings)}")
    print(f"     Identical duplicates skipped: {skipped_count}")

    if len(readings) < 10:
        print("[!] Too few readings captured to perform statistical analysis.")
        return

    # Segment into the 3 blocks
    times = np.array([r["rel_time"] for r in readings])
    amps = np.array([r["mean_amplitude"] for r in readings])

    idx_b1 = np.where(times < 30.0)[0]
    idx_b2 = np.where((times >= 30.0) & (times < 60.0))[0]
    idx_b3 = np.where(times >= 60.0)[0]

    b1_amps = amps[idx_b1] if len(idx_b1) > 0 else np.array([0.0])
    b2_amps = amps[idx_b2] if len(idx_b2) > 0 else np.array([0.0])
    b3_amps = amps[idx_b3] if len(idx_b3) > 0 else np.array([0.0])

    m1, s1 = float(np.mean(b1_amps)), float(np.std(b1_amps))
    m2, s2 = float(np.mean(b2_amps)), float(np.std(b2_amps))
    m3, s3 = float(np.mean(b3_amps)), float(np.std(b3_amps))

    # Variance / Variation metrics (crucial for gesture detection)
    diff_s1 = float(np.std(np.diff(b1_amps))) if len(b1_amps) > 1 else 0.0
    diff_s2 = float(np.std(np.diff(b2_amps))) if len(b2_amps) > 1 else 0.0
    diff_s3 = float(np.std(np.diff(b3_amps))) if len(b3_amps) > 1 else 0.0

    print("\n" + "=" * 75)
    print("STATISTICAL RESULTS PER 30-SECOND BLOCK")
    print("=" * 75)
    print(f"Block 1 [00s - 30s] (Idle 1):   Samples = {len(b1_amps):<4} | Mean Amp = {m1:6.2f} | Std = {s1:5.2f} | Diff Std = {diff_s1:5.2f}")
    print(f"Block 2 [30s - 60s] (Waving):   Samples = {len(b2_amps):<4} | Mean Amp = {m2:6.2f} | Std = {s2:5.2f} | Diff Std = {diff_s2:5.2f}")
    print(f"Block 3 [60s - 90s] (Idle 2):   Samples = {len(b3_amps):<4} | Mean Amp = {m3:6.2f} | Std = {s3:5.2f} | Diff Std = {diff_s3:5.2f}")
    print("=" * 75)

    # Save summary data
    results = {
        "total_polls": poll_count,
        "unique_readings": len(readings),
        "skipped_duplicates": skipped_count,
        "block1_idle": {"mean": round(m1, 2), "std": round(s1, 2), "diff_std": round(diff_s1, 2), "samples": len(b1_amps)},
        "block2_waving": {"mean": round(m2, 2), "std": round(s2, 2), "diff_std": round(diff_s2, 2), "samples": len(b2_amps)},
        "block3_idle": {"mean": round(m3, 2), "std": round(s3, 2), "diff_std": round(diff_s3, 2), "samples": len(b3_amps)},
        "raw_samples": [{"t": round(float(t), 2), "amp": round(float(a), 2)} for t, a in zip(times, amps)]
    }
    with open("csi_90s_experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Generate Visualization Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    plt.subplots_adjust(hspace=0.25)

    # Panel 1: Mean Subcarrier Amplitude over Time
    ax1.plot(times, amps, color="#1f77b4", linewidth=1.2, marker=".", markersize=4, alpha=0.8, label="Mean CSI Amplitude")
    ax1.axvspan(0, 30, color="gray", alpha=0.15, label="Block 1: Idle (Hands Away)")
    ax1.axvspan(30, 60, color="orange", alpha=0.25, label="Block 2: Hand Waving")
    ax1.axvspan(60, 90, color="gray", alpha=0.15, label="Block 3: Idle (Hands Away)")

    # Mean lines per block
    if len(b1_amps) > 0:
        ax1.hlines(m1, 0, 30, colors="darkblue", linestyles="--", linewidth=2, label=f"B1 Mean: {m1:.2f}")
    if len(b2_amps) > 0:
        ax1.hlines(m2, 30, 60, colors="red", linestyles="--", linewidth=2, label=f"B2 Mean: {m2:.2f}")
    if len(b3_amps) > 0:
        ax1.hlines(m3, 60, 90, colors="darkblue", linestyles="--", linewidth=2, label=f"B3 Mean: {m3:.2f}")

    ax1.set_ylabel("Mean Subcarrier Amplitude", fontsize=11, fontweight="bold")
    ax1.set_title("90-Second Fast CSI Experiment: Idle vs. Hand Wave vs. Idle", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9, ncol=2)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Rolling Temporal Standard Deviation (Motion Energy)
    rolling_w = 8
    if len(amps) >= rolling_w:
        rolling_std = [float(np.std(amps[max(0, k-rolling_w):k+1])) for k in range(len(amps))]
        ax2.plot(times, rolling_std, color="crimson", linewidth=1.5, label=f"Rolling Std Dev (w={rolling_w})")
    ax2.axvspan(0, 30, color="gray", alpha=0.15)
    ax2.axvspan(30, 60, color="orange", alpha=0.25)
    ax2.axvspan(60, 90, color="gray", alpha=0.15)
    ax2.set_xlabel("Elapsed Time (seconds)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Temporal Fluctuation (Std)", fontsize=11, fontweight="bold")
    ax2.set_title("Motion Signal / Volatility Signature", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=9)

    plot_path = "csi_90s_experiment.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[OK] Plot saved to: {plot_path}")

if __name__ == "__main__":
    run_experiment()
