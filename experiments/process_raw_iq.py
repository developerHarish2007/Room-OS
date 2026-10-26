#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone Raw CSI to Subcarrier Amplitude Converter
===================================================
Fetches live CSI data from Firebase:
https://espcsi-default-rtdb.asia-southeast1.firebasedatabase.app/csi_data.json

Converts raw_iq (256 interleaved integers = 128 pairs) to amplitude per subcarrier,
filtering out metadata headers and null/guard subcarriers.
"""

import sys
import json
import time
import math
import urllib.request
import urllib.error

URL = "https://espcsi-default-rtdb.asia-southeast1.firebasedatabase.app/csi_data.json"

# Subcarriers identified as null / guard / metadata:
# - Indices 0 and 1: Metadata header ([59, -80, 3, 0] -> length, RSSI, channel)
# - Indices 27 to 37: Exactly [0, 0] null/guard notch
# - Indices 64, 93-99: Inactive / filler subcarriers ([-1, -1] or [0, 0])
KNOWN_NULL_OR_META_INDICES = {
    0, 1,               # Header / RSSI metadata
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, # Central null band
    64,                 # Center transition marker
    93, 94, 95, 96, 97, 98, 99                  # High-band null/guard
}

def fetch_csi_data():
    """Fetches the latest csi_data.json payload from Firebase RTDB."""
    req = urllib.request.Request(URL, headers={"User-Agent": "CSI-Standalone-Reader/1.0"})
    with urllib.request.urlopen(req, timeout=5) as response:
        return json.loads(response.read().decode('utf-8'))

def parse_raw_iq_to_amplitudes(raw_iq, drop_nulls=True):
    """
    Takes 256 numbers from raw_iq, pairs them into 128 (Imag, Real) subcarriers,
    computes amplitude sqrt(I^2 + Q^2), and optionally drops null subcarriers.
    """
    total_pairs = len(raw_iq) // 2
    subcarriers = []

    for sc_idx in range(total_pairs):
        q = float(raw_iq[sc_idx * 2])
        i = float(raw_iq[sc_idx * 2 + 1])
        amp = math.sqrt(i**2 + q**2)

        is_null = (
            sc_idx in KNOWN_NULL_OR_META_INDICES or
            (abs(i) < 1e-4 and abs(q) < 1e-4) or
            (i == -1.0 and q == -1.0)
        )

        if drop_nulls and is_null:
            continue

        subcarriers.append({
            "subcarrier_idx": sc_idx,
            "real_i": i,
            "imag_q": q,
            "amplitude": round(amp, 2),
            "is_null": is_null
        })

    return subcarriers

def print_subcarrier_table(subcarriers, tag="CSI"):
    """Prints a structured table of subcarrier amplitudes."""
    print("=" * 70)
    print(f"LIVE CSI SUBCARRIER AMPLITUDES (Tag: {tag} | Active Subcarriers: {len(subcarriers)})")
    print("=" * 70)
    print(f"{'SC Index':<10} | {'Real (I)':<12} | {'Imag (Q)':<12} | {'Amplitude':<12}")
    print("-" * 55)

    # Print in 3 concise blocks or sample
    for item in subcarriers:
        print(f"SC {item['subcarrier_idx']:<7} | {item['real_i']:<12.1f} | {item['imag_q']:<12.1f} | {item['amplitude']:<12.2f}")

    print("-" * 55)
    amps = [item['amplitude'] for item in subcarriers]
    print(f"Summary: Min Amp = {min(amps):.2f} | Max Amp = {max(amps):.2f} | Mean Amp = {sum(amps)/len(amps):.2f}")
    print("=" * 70)

def main():
    stream_mode = "--stream" in sys.argv

    if stream_mode:
        print(f"[*] Streaming live subcarrier amplitudes from Firebase (Ctrl+C to stop)...")
        try:
            while True:
                data = fetch_csi_data()
                raw_iq = data.get("raw_iq", [])
                tag = data.get("tag", "CSI")
                subcarriers = parse_raw_iq_to_amplitudes(raw_iq, drop_nulls=True)
                amps = [s["amplitude"] for s in subcarriers]
                mean_amp = sum(amps)/len(amps) if amps else 0
                print(f"[LIVE] Tag: {tag} | Active SCs: {len(subcarriers)} | Mean Amp: {mean_amp:.2f} | First 5 Amps: {amps[:5]}")
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n[!] Stream stopped.")
    else:
        print(f"[*] Fetching live CSI frame from: {URL}")
        data = fetch_csi_data()
        raw_iq = data.get("raw_iq", [])
        tag = data.get("tag", "CSI")
        print(f"[*] Received payload with tag='{tag}', len(raw_iq)={len(raw_iq)}")

        subcarriers = parse_raw_iq_to_amplitudes(raw_iq, drop_nulls=True)
        print_subcarrier_table(subcarriers, tag=tag)

if __name__ == "__main__":
    main()
