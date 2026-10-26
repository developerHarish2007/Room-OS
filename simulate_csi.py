#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESP32 Fake/Simulated CSI Data Stream Generator
==============================================
Produces realistic synthetic Wi-Fi CSI data matching the exact CSV string format
from the official Espressif esp-csi library:

Line Format:
CSI_DATA,id,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,secondary_channel,local_timestamp,ant,sig_len,rx_state,len,first_word,data

Where data is a JSON array of interleaved [Imag_0, Real_0, Imag_1, Real_1, ...]
for 64 subcarriers (128 integers total) in 20MHz bandwidth.
"""

import time
import json
import numpy as np

# Header matching official esp-csi get-started examples
CSV_HEADER = [
    "type", "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth",
    "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding",
    "sgi", "noise_floor", "ampdu_cnt", "channel", "secondary_channel",
    "local_timestamp", "ant", "sig_len", "rx_state", "len", "first_word", "data"
]

class CSISimulator:
    def __init__(self, mac="94:d9:b3:80:8c:81", packet_rate_hz=50, num_subcarriers=64):
        self.mac = mac
        self.rate_hz = packet_rate_hz
        self.num_subcarriers = num_subcarriers
        self.packet_idx = 0
        self.start_time = time.time()

        # Static multipath profile: each subcarrier has a baseline amplitude and phase
        np.random.seed(42)
        self.base_amps = np.random.uniform(15.0, 35.0, size=num_subcarriers)
        self.base_phases = np.random.uniform(-np.pi, np.pi, size=num_subcarriers)

        # Standard null/guard subcarriers (zero energy)
        # 0..5, 32 (DC), 59..63
        self.null_subcarriers = [0, 1, 2, 3, 4, 5, 32, 59, 60, 61, 62, 63]
        self.base_amps[self.null_subcarriers] = 0.0

    def generate_packet(self, timestamp_us=None, inject_gesture=True, gesture_start_sec=1.5, gesture_end_sec=3.5):
        """
        Generates a single synthetic CSI_DATA packet with optional human gesture perturbation.
        """
        t = self.packet_idx / self.rate_hz
        if timestamp_us is None:
            timestamp_us = int((time.time() - self.start_time) * 1e6)

        amps = np.copy(self.base_amps)
        phases = np.copy(self.base_phases)

        # 1. Background thermal noise
        noise_amp = np.random.normal(0, 0.4, size=self.num_subcarriers)
        noise_phase = np.random.normal(0, 0.05, size=self.num_subcarriers)
        amps += noise_amp
        phases += noise_phase

        # 2. Simulate Human Gesture perturbation (e.g. hand swipe/wave at t in [gesture_start_sec, gesture_end_sec])
        # Human movement creates sinusoidal multipath fading (Doppler frequency ~2-8 Hz)
        if inject_gesture and (gesture_start_sec <= t <= gesture_end_sec):
            gesture_t = t - gesture_start_sec
            duration = gesture_end_sec - gesture_start_sec
            center_t = duration / 2.0
            # Bell-curve envelope peaking at center of gesture
            envelope = np.exp(-((gesture_t - center_t) ** 2) / 0.25)
            # Doppler modulation ~ 4 Hz
            doppler = np.sin(2 * np.pi * 4.0 * gesture_t)

            # Different subcarriers experience different multipath shifts
            for sc in range(self.num_subcarriers):
                if sc not in self.null_subcarriers:
                    sc_weight = np.sin((sc / self.num_subcarriers) * np.pi)
                    amps[sc] += 15.0 * envelope * doppler * sc_weight
                    phases[sc] += 0.9 * envelope * doppler

        # 3. Random impulse spikes (1% chance per packet) to test the Hampel filter
        if np.random.rand() < 0.015:
            spike_sc = np.random.choice([sc for sc in range(self.num_subcarriers) if sc not in self.null_subcarriers])
            amps[spike_sc] += np.random.choice([-1, 1]) * np.random.uniform(25.0, 45.0)

        # Ensure non-negative amplitudes and zero null subcarriers
        amps = np.maximum(amps, 0.0)
        amps[self.null_subcarriers] = 0.0

        # 4. Convert polar (amplitude, phase) to interleaved Cartesian [Imag_0, Real_0, Imag_1, Real_1, ...]
        raw_interleaved = []
        for sc in range(self.num_subcarriers):
            real_val = int(round(amps[sc] * np.cos(phases[sc])))
            imag_val = int(round(amps[sc] * np.sin(phases[sc])))
            raw_interleaved.append(imag_val)
            raw_interleaved.append(real_val)

        # Construct esp-csi fields
        data_json = json.dumps(raw_interleaved, separators=(',', ':'))
        rssi = int(-30 + np.random.normal(0, 1.0))
        noise_floor = -93
        channel = 13
        sig_len = 67
        data_len = len(raw_interleaved) # 128

        fields = [
            "CSI_DATA",
            str(self.packet_idx),
            self.mac,
            str(rssi),
            "11", # rate
            "1",  # sig_mode
            "6",  # mcs
            "1",  # bandwidth
            "0",  # smoothing
            "1",  # not_sounding
            "0",  # aggregation
            "1",  # stbc
            "0",  # fec_coding
            "0",  # sgi
            str(noise_floor),
            "0",  # ampdu_cnt
            str(channel),
            "2",  # secondary_channel
            str(timestamp_us),
            "0",  # ant
            str(sig_len),
            "0",  # rx_state
            str(data_len), # len (128)
            "1",  # first_word
            f'"{data_json}"' # data array
        ]

        self.packet_idx += 1
        return ",".join(fields)

def generate_simulated_dataset(num_packets=250, packet_rate_hz=50):
    """
    Generates a list of simulated CSI_DATA lines.
    """
    sim = CSISimulator(packet_rate_hz=packet_rate_hz)
    lines = [",".join(CSV_HEADER)]
    for _ in range(num_packets):
        lines.append(sim.generate_packet())
    return lines
