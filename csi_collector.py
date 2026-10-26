#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESP32 CSI Serial Collector and Logger
====================================
Collects CSI packets from:
  1. Real ESP32 board over Serial (COM port, default 921600 baud)
  2. Simulated CSI stream (--simulate flag, perfect when no hardware is connected)

Features:
  - Logs raw CSI lines to CSV
  - Optional real-time or post-run preprocessing (amplitude extraction, null removal,
    spike filtering, smoothing, and normalization)
  - Auto-detection of available COM ports
"""

import sys
import os
import csv
import io
import time
import argparse
import numpy as np

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

from csi_preprocessor import (
    parse_csi_data_field,
    extract_amplitude,
    remove_null_subcarriers,
    filter_spikes,
    smooth_csi,
    normalize_csi,
    preprocess_csi_pipeline
)
from simulate_csi import CSISimulator, CSV_HEADER
from csi_motion_detector import CSIMotionDetector
from json_publisher import MotionStatePublisher

def list_available_ports():
    """Returns a list of detected serial COM ports."""
    if not SERIAL_AVAILABLE:
        print("[!] Warning: 'pyserial' is not installed.")
        return []
    ports = serial.tools.list_ports.comports()
    return ports

def auto_detect_esp32_port():
    """Attempts to automatically find an ESP32 USB-to-UART bridge."""
    ports = list_available_ports()
    for p in ports:
        desc = (p.description or "").lower()
        hwid = (p.hwid or "").lower()
        if any(keyword in desc or keyword in hwid for keyword in ["cp210", "ch340", "ch341", "ftdi", "uart", "esp32"]):
            return p.device
    if len(ports) > 0:
        return ports[0].device
    return None

def collect_from_stream(line_generator, output_raw_csv, output_clean_csv=None, max_packets=None, max_duration=None, enable_detector=True, serial_port_obj=None, calib_sec=5.0):
    """
    Core loop reading CSI_DATA lines from either serial or simulator.
    """
    raw_file = open(output_raw_csv, mode="w", newline="", encoding="utf-8")
    raw_writer = csv.writer(raw_file)
    raw_writer.writerow(CSV_HEADER)

    packet_count = 0
    start_time = time.time()
    last_print = start_time

    complex_frames = []
    metadata_rows = []

    detector = None
    if enable_detector:
        detector = CSIMotionDetector(
            fps=50,
            calibration_sec=calib_sec,
            serial_port_obj=serial_port_obj,
            save_dir="gesture_samples"
        )
        # Attach JSON state publisher
        MotionStatePublisher(detector, verbose=False)

    print(f"[*] Starting CSI collection -> Logging to '{output_raw_csv}'")
    if enable_detector:
        print(f"[*] Live Motion Detector ACTIVE (Calibrating first {calib_sec:.1f}s)")
        print(f"    - Segmented gesture examples will be saved to: 'gesture_samples/'")
        print(f"    - Live JSON state published to 'latest_motion_state.json' every 0.5s")
    print("[*] Press Ctrl+C at any time to finish and run preprocessing.")

    try:
        for line in line_generator:
            if not line:
                continue

            # Strip carriage returns and leading byte artifacts
            clean_line = line.strip().lstrip("b'").rstrip("'")
            if not clean_line.startswith("CSI_DATA"):
                continue

            # Parse CSV row
            row_reader = csv.reader(io.StringIO(clean_line))
            row = next(row_reader)

            # esp-csi row should have 25 columns
            if len(row) < 25:
                continue

            raw_writer.writerow(row)
            packet_count += 1

            # Extract complex subcarriers from data column (index -1)
            try:
                complex_sc = parse_csi_data_field(row[-1])
                complex_frames.append(complex_sc)
                # Keep lightweight metadata: packet id, rssi, timestamp
                metadata_rows.append({
                    "packet_id": row[1],
                    "mac": row[2],
                    "rssi": row[3],
                    "timestamp": row[18]
                })

                # Process live motion detector
                if detector:
                    amp = np.abs(complex_sc)
                    clean_vec, _ = remove_null_subcarriers(amp)
                    res = detector.process_frame(clean_vec[0], timestamp=time.time())
                    if res.get("saved_gesture"):
                        print(f"\n[!] Motion Event Finished -> Saved gesture window to: {res['saved_gesture']}")

            except Exception as e:
                # Malformed JSON in corrupted serial transmission
                continue

            now = time.time()
            if now - last_print >= 1.0:
                elapsed = now - start_time
                fps = packet_count / elapsed if elapsed > 0 else 0
                rssi_disp = row[3] if len(row) > 3 else "N/A"
                det_status = f" | State: {detector.state} (Score: {detector.current_score:.2f})" if detector else ""
                print(f"  [+] Packets: {packet_count} | Rate: {fps:.1f} pkt/s | RSSI: {rssi_disp} dBm{det_status}")
                last_print = now

            if max_packets and packet_count >= max_packets:
                print(f"[!] Reached target packet count of {max_packets}.")
                break
            if max_duration and (now - start_time) >= max_duration:
                print(f"[!] Reached target duration of {max_duration} seconds.")
                break

    except KeyboardInterrupt:
        print("\n[!] User interrupted collection (Ctrl+C). Wrapping up...")
    finally:
        raw_file.close()

    total_time = time.time() - start_time
    avg_fps = packet_count / total_time if total_time > 0 else 0
    print(f"\n[OK] Completed: Captured {packet_count} packets in {total_time:.2f}s (Avg {avg_fps:.1f} Hz)")

    # Preprocessing
    if output_clean_csv and len(complex_frames) > 0:
        print(f"[*] Running signal preprocessing pipeline on {len(complex_frames)} frames...")
        complex_matrix = np.array(complex_frames)

        # Run pipeline
        preprocessed_amp, meta = preprocess_csi_pipeline(
            complex_matrix,
            remove_nulls=True,
            filter_outliers=True,
            smooth=True,
            normalize=True,
            smooth_method="savgol",
            norm_method="zscore"
        )

        active_indices = meta.get("active_subcarrier_indices", list(range(preprocessed_amp.shape[1])))
        header = ["packet_id", "rssi", "timestamp"] + [f"subcarrier_{sc}" for sc in active_indices]

        with open(output_clean_csv, mode="w", newline="", encoding="utf-8") as f_clean:
            clean_writer = csv.writer(f_clean)
            clean_writer.writerow(header)
            for idx in range(len(preprocessed_amp)):
                meta_item = metadata_rows[idx]
                row_vals = [
                    meta_item["packet_id"],
                    meta_item["rssi"],
                    meta_item["timestamp"]
                ] + list(preprocessed_amp[idx, :])
                clean_writer.writerow(row_vals)

        print(f"[OK] Preprocessed CSI features written to: '{output_clean_csv}'")
        print(f"    - Active subcarriers retained: {meta.get('active_subcarriers_count')} (nulls/pilots removed)")
        print(f"    - Feature shape: {preprocessed_amp.shape} (packets x subcarriers)")

    return packet_count

def run_serial_collector(port, baud, raw_csv, clean_csv, max_packets=None, max_duration=None):
    """Reads live from a hardware ESP32 board over serial."""
    if not SERIAL_AVAILABLE:
        print("[!] Error: 'pyserial' is required for serial communication. Install via 'pip install pyserial'.")
        return

    print(f"[*] Connecting to ESP32 on port '{port}' at {baud} baud...")
    try:
        ser = serial.Serial(port=port, baudrate=baud, bytesize=8, parity='N', stopbits=1, timeout=1.0)
    except Exception as e:
        print(f"[!] Error opening serial port '{port}': {e}")
        return

    def serial_generator():
        while True:
            line_bytes = ser.readline()
            if not line_bytes:
                continue
            yield line_bytes.decode("utf-8", errors="ignore")

    try:
        collect_from_stream(serial_generator(), raw_csv, clean_csv, max_packets, max_duration, serial_port_obj=ser)
    finally:
        ser.close()

def run_simulator_collector(rate_hz, raw_csv, clean_csv, max_packets=None, max_duration=None):
    """Reads synthetic CSI packets from CSISimulator."""
    print(f"[*] Running in SIMULATOR mode (Rate: {rate_hz} Hz, generating synthetic gesture data)")
    sim = CSISimulator(packet_rate_hz=rate_hz)
    interval = 1.0 / rate_hz

    def sim_generator():
        while True:
            t0 = time.time()
            packet_str = sim.generate_packet(inject_gesture=True)
            yield packet_str
            # Sleep to match physical packet transmission rate
            elapsed = time.time() - t0
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    collect_from_stream(sim_generator(), raw_csv, clean_csv, max_packets, max_duration)

def main():
    parser = argparse.ArgumentParser(description="ESP32 CSI Serial Collector and Preprocessor")
    parser.add_argument("-p", "--port", type=str, default=None, help="Serial port (e.g. COM3 or /dev/ttyUSB0)")
    parser.add_argument("-b", "--baud", type=int, default=921600, help="Baud rate (default 921600 for esp-csi)")
    parser.add_argument("-o", "--output", type=str, default="csi_raw_log.csv", help="Output raw CSV file path")
    parser.add_argument("-c", "--clean-output", type=str, default="csi_preprocessed.csv", help="Output preprocessed CSV path")
    parser.add_argument("-s", "--simulate", action="store_true", help="Run simulated CSI stream instead of serial port")
    parser.add_argument("-n", "--packets", type=int, default=None, help="Target number of packets to collect")
    parser.add_argument("-t", "--duration", type=float, default=None, help="Target duration in seconds")
    parser.add_argument("--rate", type=int, default=50, help="Simulation packet rate in Hz (default: 50)")
    parser.add_argument("--list-ports", action="store_true", help="List all available COM ports and exit")

    args = parser.parse_args()

    if args.list_ports:
        ports = list_available_ports()
        if not ports:
            print("[!] No active COM ports found.")
        else:
            print("[*] Available Serial Ports:")
            for p in ports:
                print(f"    - {p.device}: {p.description}")
        return

    if args.simulate:
        run_simulator_collector(args.rate, args.output, args.clean_output, args.packets, args.duration)
        return

    # Real hardware mode
    target_port = args.port or auto_detect_esp32_port()
    if not target_port:
        print("[!] No serial port specified and no ESP32 detected.")
        print("    Tip: Use '--simulate' to test without hardware, or specify '--port COMx'.")
        return

    run_serial_collector(target_port, args.baud, args.output, args.clean_output, args.packets, args.duration)

if __name__ == "__main__":
    main()
