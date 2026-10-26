#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESP32 CSI Local WebSocket Server (Prompt B Integration)
======================================================
Reads CSI packets from ESP32 serial port (or simulated stream),
drops null/guard subcarriers, computes per-subcarrier amplitudes,
computes a rolling 0.5s standard-deviation motion score,
and streams live packets to browser dashboard over WebSocket (ws://localhost:8765).
"""

import sys
import re
import json
import time
import math
import asyncio
import argparse
from collections import deque

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

# Known null, guard, DC center, and metadata subcarrier indices
KNOWN_NULL_OR_META_INDICES = {
    0, 1,                                        # Header / RSSI metadata
    27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,  # Central null band
    64,                                          # Center transition marker
    93, 94, 95, 96, 97, 98, 99                   # High-band null/guard
}

def list_serial_ports():
    if not SERIAL_AVAILABLE:
        return []
    return [p.device for p in serial.tools.list_ports.comports()]

def parse_raw_iq_from_line(line):
    """Extracts raw I/Q integers and RSSI from a CSI_DATA line."""
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

    rssi = None
    parts = line.split(',')
    if len(parts) >= 4:
        try:
            rssi = int(parts[3])
        except (ValueError, IndexError):
            pass

    return raw_iq, rssi

def process_iq_to_amplitudes(raw_iq, drop_nulls=True):
    """Pairs raw_iq into (Imag, Real) subcarriers, drops nulls, returns amplitudes."""
    num_pairs = len(raw_iq) // 2
    active_amplitudes = []

    for sc_idx in range(num_pairs):
        q = float(raw_iq[sc_idx * 2])
        i = float(raw_iq[sc_idx * 2 + 1])

        is_null = (
            sc_idx in KNOWN_NULL_OR_META_INDICES or
            (abs(i) < 1e-4 and abs(q) < 1e-4) or
            (i == -1.0 and q == -1.0)
        )

        if drop_nulls and is_null:
            continue

        amp = math.sqrt(i * i + q * q)
        active_amplitudes.append(round(amp, 2))

    return active_amplitudes

class CSIServerEngine:
    def __init__(self, port=None, baudrate=921600, simulate=False):
        self.port = port
        self.baudrate = baudrate
        self.simulate = simulate
        self.clients = set()
        
        # Sliding buffer for rolling 0.5s window
        self.history_window = deque()  # stores (timestamp, [amplitudes])
        self.window_duration = 0.5     # 0.5 second window
        
        # Real PPS metrics
        self.packet_timestamps = deque()
        self.current_pps = 0.0
        self.total_packets = 0
        
        # Idle Calibration State
        self.is_calibrating = False
        self.calib_samples = []
        self.baseline_mean = 0.0
        self.baseline_std = 0.0
        self.motion_threshold = 2.5
        
        # Synthetic simulator state
        self.sim_phase = 0.0
        self.sim_gesture_counter = 0

    def compute_motion_score(self, current_amps):
        """
        Computes motion score as average over subcarriers of the standard deviation
        across the last 0.5 seconds (Prompt B specification), with AGC normalization.
        """
        now = time.time()
        self.history_window.append((now, current_amps))

        # Evict frames older than 0.5 seconds
        while self.history_window and self.history_window[0][0] < (now - self.window_duration):
            self.history_window.popleft()

        num_frames = len(self.history_window)
        if num_frames < 3:
            return 0.0

        num_subcarriers = len(current_amps)
        if num_subcarriers == 0:
            return 0.0

        # Calculate mean amplitude per frame to normalize against AGC shifts
        norm_frames = []
        for _, amps in self.history_window:
            if len(amps) != num_subcarriers:
                continue
            f_mean = sum(amps) / len(amps) if amps else 1.0
            if f_mean < 1e-3:
                f_mean = 1.0
            norm_frames.append([a / f_mean for a in amps])

        if len(norm_frames) < 3:
            return 0.0

        N = len(norm_frames)
        total_std = 0.0
        for sc in range(num_subcarriers):
            sc_mean = sum(norm_frames[f][sc] for f in range(N)) / N
            sc_var = sum((norm_frames[f][sc] - sc_mean) ** 2 for f in range(N)) / N
            total_std += math.sqrt(sc_var)

        # Scale by 100 for intuitive score (e.g. 0.2 idle, 3.0+ for hand wave)
        motion_score = (total_std / num_subcarriers) * 100.0
        return round(motion_score, 2)

    def record_packet(self):
        now = time.time()
        self.total_packets += 1
        self.packet_timestamps.append(now)
        while self.packet_timestamps and self.packet_timestamps[0] < now - 1.0:
            self.packet_timestamps.popleft()

        if len(self.packet_timestamps) > 1:
            span = self.packet_timestamps[-1] - self.packet_timestamps[0]
            self.current_pps = round((len(self.packet_timestamps) - 1) / max(0.001, span), 1)
        else:
            self.current_pps = 1.0

    async def register_client(self, websocket):
        self.clients.add(websocket)
        print(f"[+] Client connected from {websocket.remote_address}. Active clients: {len(self.clients)}")
        # Send initial handshake with configuration
        handshake = {
            "type": "init",
            "source": "simulated" if self.simulate else f"serial:{self.port}",
            "baudrate": self.baudrate,
            "threshold": self.motion_threshold,
            "timestamp": time.time()
        }
        await websocket.send(json.dumps(handshake))

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    action = data.get("action")
                    if action == "calibrate_idle":
                        duration = data.get("duration", 4.0)
                        asyncio.create_task(self.run_idle_calibration(websocket, duration))
                    elif action == "set_threshold":
                        self.motion_threshold = float(data.get("threshold", 2.5))
                        print(f"[*] Updated threshold to: {self.motion_threshold}")
                except Exception as e:
                    print(f"[!] Message parse error: {e}")
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            print(f"[-] Client disconnected. Active clients: {len(self.clients)}")

    async def broadcast(self, payload):
        if not self.clients:
            return
        msg = json.dumps(payload)
        stale = set()
        for client in self.clients:
            try:
                await client.send(msg)
            except Exception:
                stale.add(client)
        for client in stale:
            self.clients.discard(client)

    async def run_idle_calibration(self, target_client, duration=4.0):
        print(f"[*] Starting {duration}s idle baseline calibration...")
        self.is_calibrating = True
        self.calib_samples = []
        start_time = time.time()

        while time.time() - start_time < duration:
            await asyncio.sleep(0.1)

        self.is_calibrating = False
        if len(self.calib_samples) >= 10:
            mean = sum(self.calib_samples) / len(self.calib_samples)
            variance = sum((x - mean) ** 2 for x in self.calib_samples) / len(self.calib_samples)
            std = math.sqrt(variance)
            new_thresh = round(max(0.8, mean + 3.0 * std), 2)
            self.motion_threshold = new_thresh
            print(f"[OK] Calibration complete: Idle Mean={mean:.2f}, Std={std:.2f} -> Threshold={new_thresh:.2f}")

            res = {
                "type": "calibration_result",
                "idle_mean": round(mean, 2),
                "idle_std": round(std, 2),
                "recommended_threshold": new_thresh
            }
            await self.broadcast(res)
        else:
            print("[!] Insufficient packets for calibration.")

    def generate_simulated_frame(self):
        """Generates realistic synthetic 107-subcarrier CSI packet."""
        self.sim_phase += 0.1
        self.sim_gesture_counter = (self.sim_gesture_counter + 1) % 400

        # Simulate hand wave motion periodically between ticks 150-250
        is_moving = 150 <= self.sim_gesture_counter <= 250
        motion_factor = 3.5 if is_moving else 0.08

        base_amps = []
        for sc in range(107):
            # Static multipath profile + thermal noise
            profile = 22.0 + 8.0 * math.sin(sc * 0.12)
            noise = (math.sin(sc * 2.3 + self.sim_phase) * 0.4)
            # Multipath interference distortion across subcarriers
            wave_perturbation = math.sin(self.sim_phase * 2.5 + sc * 0.3) * motion_factor
            base_amps.append(round(max(2.0, profile + noise + wave_perturbation), 2))

        return base_amps, -44 + int(math.sin(self.sim_phase * 0.5) * 3)

    async def stream_loop(self):
        """Producer loop reading serial port or running simulation."""
        ser = None
        if not self.simulate:
            if not self.port:
                ports = list_serial_ports()
                if ports:
                    self.port = ports[0]
                    print(f"[*] Auto-selected first available COM port: {self.port}")
                else:
                    print("[!] No COM ports detected. Switching automatically to Simulation Mode.")
                    self.simulate = True

            if not self.simulate:
                try:
                    ser = serial.Serial(self.port, self.baudrate, timeout=0.05)
                    print(f"[OK] Opened serial port {self.port} at {self.baudrate} baud.")
                except Exception as e:
                    print(f"[!] Could not open serial port '{self.port}': {e}")
                    print("[*] Falling back to high-fidelity Simulation Mode.")
                    self.simulate = True

        print(f"[*] Live Stream running in {'SIMULATION' if self.simulate else 'HARDWARE SERIAL'} mode.")

        while True:
            t0 = time.time()
            active_amps = None
            rssi = -45

            if self.simulate:
                active_amps, rssi = self.generate_simulated_frame()
                await asyncio.sleep(0.016)  # ~60 FPS simulation
            else:
                try:
                    line_bytes = ser.readline()
                    if not line_bytes:
                        await asyncio.sleep(0.002)
                        continue
                    line = line_bytes.decode("utf-8", errors="ignore").strip()
                    if "CSI_DATA" not in line:
                        continue
                    raw_iq, parsed_rssi = parse_raw_iq_from_line(line)
                    if not raw_iq or len(raw_iq) < 20:
                        continue
                    active_amps = process_iq_to_amplitudes(raw_iq, drop_nulls=True)
                    if parsed_rssi is not None:
                        rssi = parsed_rssi
                except Exception as err:
                    print(f"[!] Serial read error: {err}")
                    await asyncio.sleep(0.05)
                    continue

            if active_amps and len(active_amps) > 0:
                self.record_packet()
                motion_score = self.compute_motion_score(active_amps)

                if self.is_calibrating:
                    self.calib_samples.append(motion_score)

                # Send packet payload to connected web clients
                payload = {
                    "type": "csi_frame",
                    "timestamp": round(t0, 4),
                    "amplitudes": active_amps,
                    "mean_amp": round(sum(active_amps) / len(active_amps), 2),
                    "rssi": rssi,
                    "motion_score": motion_score,
                    "packet_rate": self.current_pps,
                    "source": "simulated" if self.simulate else "serial"
                }
                await self.broadcast(payload)

async def main():
    parser = argparse.ArgumentParser(description="ESP32 CSI WebSocket Streaming Server")
    parser.add_argument("-p", "--port", type=str, default=None, help="Serial port (e.g. COM3)")
    parser.add_argument("-b", "--baud", type=int, default=921600, help="Baud rate (default 921600)")
    parser.add_argument("-s", "--simulate", action="store_true", help="Run synthetic CSI simulation")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket port (default 8765)")
    args = parser.parse_args()

    engine = CSIServerEngine(port=args.port, baudrate=args.baud, simulate=args.simulate)

    print("=" * 65)
    print("ESP32 CSI WEBSOCKET STREAMING SERVER (PROMPT B)")
    print(f"WebSocket URL: ws://localhost:{args.ws_port}")
    print("=" * 65)

    ws_server = await websockets.serve(engine.register_client, "0.0.0.0", args.ws_port)
    await engine.stream_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[!] Server stopped by user.")
