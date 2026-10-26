#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESP32 CSI Serial Capture & Real-Time PPS Monitor
================================================
Reads raw CSI_DATA lines from the ESP32 receiver's serial port.
- Parses interleaved raw I/Q array
- Filters out null, guard, metadata, and filler subcarriers
- Computes amplitude: sqrt(I^2 + Q^2) per active subcarrier
- Calculates and prints real received packets per second (PPS / Hz) in real-time
"""

import sys
import re
import time
import math
import argparse
from collections import deque

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# Known null, guard, DC center, and metadata subcarrier indices
KNOWN_NULL_OR_META_INDICES = {
    0, 1,                                        # Header / RSSI metadata
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,  # Central null band
    64,                                          # Center transition marker
    93, 94, 95, 96, 97, 98, 99                   # High-band null/guard
}

def list_serial_ports():
    """Lists all currently available serial/COM ports."""
    if not SERIAL_AVAILABLE:
        return []
    return list(serial.tools.list_ports.comports())

def prompt_for_com_port():
    """Lists available ports and asks user to specify the COM port."""
    ports = list_serial_ports()
    print("\n" + "=" * 60)
    print("ESP32 CSI SERIAL CAPTURE - PORT SELECTION")
    print("=" * 60)
    
    if ports:
        print("Available COM Ports Detected:")
        for idx, p in enumerate(ports, 1):
            desc = p.description or "Unknown device"
            print(f"  [{idx}] {p.device} - {desc}")
    else:
        print("[!] No active COM ports detected by the operating system.")
        print("    Please connect your ESP32 receiver board via USB.")

    print("-" * 60)
    while True:
        try:
            user_input = input("Enter COM port (e.g. COM3 or COM4): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted by user.")
            sys.exit(0)

        if not user_input:
            print("Port cannot be empty. Please enter a valid port name (e.g. COM3).")
            continue

        # If user entered an index number from the list
        if user_input.isdigit() and ports:
            idx = int(user_input) - 1
            if 0 <= idx < len(ports):
                return ports[idx].device

        return user_input.upper() if user_input.lower().startswith("com") else user_input

def parse_raw_iq_from_line(line):
    """
    Extracts raw I/Q integers from a CSI_DATA serial line.
    ESP-CSI formats line like:
    CSI_DATA,<id>,<mac>,<rssi>,...,len,first_word,"[q0,i0,q1,i1,...]"
    """
    bracket_match = re.search(r'\[(.*?)\]', line)
    if not bracket_match:
        return None, None

    raw_str = bracket_match.group(1).strip()
    if not raw_str:
        return None, None

    try:
        raw_iq = [int(x.strip()) for x in raw_str.split(',') if x.strip()]
    except ValueError:
        return None, None

    # Parse RSSI if available (typically 4th CSV column)
    rssi = None
    parts = line.split(',')
    if len(parts) >= 4:
        try:
            rssi = int(parts[3])
        except (ValueError, IndexError):
            pass

    return raw_iq, rssi

def process_iq_to_amplitudes(raw_iq, drop_nulls=True):
    """
    Pairs raw_iq into (Imag, Real) subcarriers, drops null/guard subcarriers,
    and returns list of float amplitudes.
    """
    num_pairs = len(raw_iq) // 2
    active_amplitudes = []

    for sc_idx in range(num_pairs):
        q = float(raw_iq[sc_idx * 2])
        i = float(raw_iq[sc_idx * 2 + 1])

        # Check if subcarrier is null, guard, or unused filler
        is_null = (
            sc_idx in KNOWN_NULL_OR_META_INDICES or
            (abs(i) < 1e-4 and abs(q) < 1e-4) or
            (i == -1.0 and q == -1.0)
        )

        if drop_nulls and is_null:
            continue

        amp = math.sqrt(i * i + q * q)
        active_amplitudes.append(amp)

    return active_amplitudes

def run_serial_capture(port, baudrate=921600):
    """Main loop reading from the receiver's serial port."""
    if not SERIAL_AVAILABLE:
        print("[!] Error: 'pyserial' package is not available. Install it with: pip install pyserial")
        sys.exit(1)

    print(f"\n[*] Connecting to ESP32 on port '{port}' at {baudrate} baud...")
    try:
        ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1.0
        )
    except serial.SerialException as e:
        print(f"[!] Failed to open serial port '{port}': {e}")
        print("    Tips: Check device connection, close any conflicting serial monitors, and verify COM port name.")
        sys.exit(1)

    print(f"[OK] Successfully connected to '{port}'. Waiting for CSI_DATA stream...")
    print("     (Press Ctrl+C to stop capture)\n")
    print(f"{'Packet':<8} | {'Real Rate':<12} | {'Active SC':<10} | {'Mean Amp':<10} | {'RSSI':<8} | {'Preview Amps (first 5)'}")
    print("-" * 80)

    packet_counter = 0
    recent_timestamps = deque()  # Sliding window of timestamps for instantaneous rate calculation
    last_stat_print = time.time()
    current_pps = 0.0

    try:
        while True:
            line_bytes = ser.readline()
            if not line_bytes:
                continue

            try:
                line = line_bytes.decode("utf-8", errors="ignore").strip()
            except Exception:
                continue

            # Only process lines starting with or containing CSI_DATA
            if "CSI_DATA" not in line:
                continue

            now = time.time()
            raw_iq, rssi = parse_raw_iq_from_line(line)
            if not raw_iq or len(raw_iq) < 20:
                continue

            packet_counter += 1
            recent_timestamps.append(now)

            # Keep only timestamps within the last 1.0 second window for accurate PPS
            while recent_timestamps and recent_timestamps[0] < now - 1.0:
                recent_timestamps.popleft()

            # Calculate actual packets per second
            if len(recent_timestamps) > 1:
                time_span = recent_timestamps[-1] - recent_timestamps[0]
                if time_span > 0:
                    current_pps = (len(recent_timestamps) - 1) / time_span
            else:
                current_pps = 1.0

            # Process Amplitudes
            active_amplitudes = process_iq_to_amplitudes(raw_iq, drop_nulls=True)
            mean_amp = (sum(active_amplitudes) / len(active_amplitudes)) if active_amplitudes else 0.0
            rssi_str = f"{rssi} dBm" if rssi is not None else "--"
            preview = ", ".join(f"{a:.1f}" for a in active_amplitudes[:5]) if active_amplitudes else "--"

            # Print packet status (throttled slightly to ~10 prints/sec for clean terminal readability)
            if now - last_stat_print >= 0.1 or packet_counter % 10 == 0:
                last_stat_print = now
                print(f"#{packet_counter:<7} | {current_pps:>6.1f} PPS  | {len(active_amplitudes):>4} SC    | {mean_amp:>8.2f}   | {rssi_str:>8} | [{preview}, ...]")

    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("CAPTURE SUMMARY")
        print("=" * 60)
        print(f"Total packets captured: {packet_counter}")
        print(f"Final packet rate:      {current_pps:.1f} packets/second")
        print("=" * 60)
    finally:
        ser.close()
        print(f"Serial port '{port}' closed cleanly.")

def main():
    parser = argparse.ArgumentParser(
        description="ESP32 CSI Serial Capture & Real-Time PPS Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-p", "--port",
        type=str,
        default=None,
        help="Serial COM port for ESP32 receiver (e.g. COM3, COM4). If omitted, the script prompts for it."
    )
    parser.add_argument(
        "-b", "--baud",
        type=int,
        default=921600,
        help="Serial baud rate (default: 921600 for ESP-CSI)"
    )

    args = parser.parse_args()

    port = args.port
    if not port:
        port = prompt_for_com_port()

    run_serial_capture(port=port, baudrate=args.baud)

if __name__ == "__main__":
    main()
