#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSI Motion & Anomaly Detector with Gesture Window Segmentation
=============================================================
Features:
1. Baseline calibration on initial idle period (mean & std).
2. Rolling subcarrier variance/variation motion score.
3. Dual-threshold hysteresis (T_high, T_low) + debounce counter to prevent flicker.
4. Automatic gesture window cut-out (~1-2 seconds) saved to disk for future classifier.
5. Periodic (0.5s) JSON state publisher ready for teammate integration (e.g. Firebase).
6. Optional LED blink serial command trigger for hardware receiver.
"""

import os
import json
import time
import threading
from collections import deque
import numpy as np

class CSIMotionDetector:
    def __init__(self,
                 fps=50,
                 calibration_sec=10.0,
                 rolling_window_sec=0.4,
                 k_std_high=3.5,
                 k_std_low=1.8,
                 debounce_trigger_frames=4,
                 debounce_release_frames=12,
                 gesture_window_sec=1.5,
                 save_dir="gesture_samples",
                 serial_port_obj=None,
                 json_publish_interval=0.5):
        """
        fps: packets per second from CSI receiver (default 50 Hz).
        calibration_sec: duration of initial quiet state to learn baseline (default 10s).
        rolling_window_sec: window over which subcarrier variance is computed.
        k_std_high: multiplier above baseline std to trigger MOTION.
        k_std_low: multiplier above baseline std to return to IDLE (hysteresis).
        debounce_trigger_frames: frames score must stay > T_high before flipping to MOTION.
        debounce_release_frames: frames score must stay < T_low before flipping to IDLE.
        gesture_window_sec: duration of segmented gesture example cut-out (~1.5s).
        save_dir: directory where segmented gesture examples are saved.
        serial_port_obj: optional opened serial.Serial instance to send LED blink command.
        json_publish_interval: interval in seconds to publish periodic JSON status (0.5s).
        """
        self.fps = fps
        self.calib_frames_target = int(calibration_sec * fps)
        self.rolling_window_len = max(5, int(rolling_window_sec * fps))
        self.k_high = k_std_high
        self.k_low = k_std_low
        self.debounce_trig_limit = debounce_trigger_frames
        self.debounce_rel_limit = debounce_release_frames
        self.gesture_window_frames = int(gesture_window_sec * fps)
        self.pre_trigger_frames = int(0.3 * fps) # 300ms pre-gesture context
        self.save_dir = save_dir
        self.serial_port = serial_port_obj
        self.json_interval = json_publish_interval

        os.makedirs(self.save_dir, exist_ok=True)

        # State variables
        self.state = "CALIBRATING"
        self.current_score = 0.0
        self.baseline_mean = 0.0
        self.baseline_std = 0.0
        self.threshold_high = 0.0
        self.threshold_low = 0.0

        # Debounce counters
        self.trig_counter = 0
        self.rel_counter = 0

        # Buffers
        self.calib_scores = []
        self.rolling_frames = deque(maxlen=self.rolling_window_len)
        self.history_buffer = deque(maxlen=self.gesture_window_frames + self.pre_trigger_frames)
        self.active_gesture_buffer = []

        # Segmentation tracking
        self.gesture_sample_count = 0
        self.is_recording_gesture = False

        # Listeners / callbacks for external teammates (Firebase / Web)
        self.listeners = []
        self.last_publish_time = time.time()
        self._lock = threading.Lock()

    def add_listener(self, callback):
        """
        Registers a callback function that receives payload dict whenever
        published or state changes: callback(payload_dict)
        """
        self.listeners.append(callback)

    def _compute_motion_score(self, current_frame):
        """
        Calculates a motion score S(t) by measuring the rolling dynamic variation
        across active subcarriers. When people move, multipath fluctuations cause
        rapid temporal variance compared to static background noise.
        """
        self.rolling_frames.append(current_frame)
        if len(self.rolling_frames) < 3:
            return 0.0

        window_arr = np.array(self.rolling_frames) # (W, num_subcarriers)
        # Compute standard deviation across time for each subcarrier, then average across subcarriers
        subcarrier_stds = np.std(window_arr, axis=0)
        # Also compute consecutive frame diffs
        frame_diffs = np.mean(np.abs(np.diff(window_arr, axis=0)))

        score = float(np.mean(subcarrier_stds) + 0.5 * frame_diffs)
        return score

    def process_frame(self, preprocessed_subcarrier_vec, timestamp=None):
        """
        Feeds one preprocessed subcarrier amplitude vector (shape: N_subcarriers)
        Returns: dict with current state, motion_score, and any segmented gesture info.
        """
        if timestamp is None:
            timestamp = time.time()

        vec = np.array(preprocessed_subcarrier_vec, dtype=np.float64)
        score = self._compute_motion_score(vec)
        self.current_score = score
        self.history_buffer.append(vec)

        event_info = None

        with self._lock:
            # 1. Calibration phase
            if self.state == "CALIBRATING":
                self.calib_scores.append(score)
                if len(self.calib_scores) >= self.calib_frames_target:
                    # Calibrate baseline statistics
                    scores_arr = np.array(self.calib_scores)
                    self.baseline_mean = float(np.mean(scores_arr))
                    self.baseline_std = float(np.std(scores_arr))
                    if self.baseline_std < 1e-4:
                        self.baseline_std = 0.05 # Prevent zero division

                    self.threshold_high = self.baseline_mean + self.k_high * self.baseline_std
                    self.threshold_low = self.baseline_mean + self.k_low * self.baseline_std
                    self.state = "IDLE"
                    event_info = "CALIBRATION_COMPLETE"
                    self._publish_state(force=True)

            # 2. Operational state machine (Hysteresis & Debounce)
            elif self.state == "IDLE":
                if score >= self.threshold_high:
                    self.trig_counter += 1
                    if self.trig_counter >= self.debounce_trig_limit:
                        # State transition: IDLE -> MOTION
                        self.state = "MOTION"
                        self.trig_counter = 0
                        self.rel_counter = 0
                        self.is_recording_gesture = True
                        self.active_gesture_buffer = list(self.history_buffer)
                        event_info = "MOTION_START"

                        # Send hardware LED blink command if serial connected
                        self._trigger_hardware_led(True)
                        self._publish_state(force=True)
                else:
                    self.trig_counter = max(0, self.trig_counter - 1)

            elif self.state == "MOTION":
                if self.is_recording_gesture:
                    self.active_gesture_buffer.append(vec)

                if score <= self.threshold_low:
                    self.rel_counter += 1
                    if self.rel_counter >= self.debounce_rel_limit:
                        # State transition: MOTION -> IDLE
                        self.state = "IDLE"
                        self.rel_counter = 0
                        self.trig_counter = 0
                        event_info = "MOTION_END"

                        # Turn off / un-trigger hardware LED
                        self._trigger_hardware_led(False)

                        # Cut out and save the segmented gesture window
                        saved_path = self._save_segmented_gesture()
                        self.is_recording_gesture = False
                        self.active_gesture_buffer = []
                        self._publish_state(force=True)
                        return {
                            "state": self.state,
                            "score": score,
                            "timestamp": timestamp,
                            "event": event_info,
                            "saved_gesture": saved_path
                        }
                else:
                    self.rel_counter = max(0, self.rel_counter - 1)

        # Periodic JSON heartbeat publish check
        now = time.time()
        if now - self.last_publish_time >= self.json_interval:
            self._publish_state(force=False)

        return {
            "state": self.state,
            "score": score,
            "timestamp": timestamp,
            "event": event_info,
            "saved_gesture": None
        }

    def _save_segmented_gesture(self):
        """
        Extracts the segmented 1-2 second gesture window, pads/trims to target length,
        and saves as both a numpy array (.npy) and metadata json.
        """
        if len(self.active_gesture_buffer) < 10:
            return None

        gesture_matrix = np.array(self.active_gesture_buffer)
        # Target length around gesture_window_frames (~1.5s)
        if len(gesture_matrix) > self.gesture_window_frames:
            # Crop to center/peak window
            gesture_matrix = gesture_matrix[:self.gesture_window_frames]

        self.gesture_sample_count += 1
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        filename_base = f"gesture_{timestamp_str}_{self.gesture_sample_count:03d}"
        npy_path = os.path.join(self.save_dir, f"{filename_base}.npy")
        json_path = os.path.join(self.save_dir, f"{filename_base}.json")

        np.save(npy_path, gesture_matrix)

        meta = {
            "sample_id": self.gesture_sample_count,
            "timestamp": time.time(),
            "frames": gesture_matrix.shape[0],
            "subcarriers": gesture_matrix.shape[1],
            "duration_sec": gesture_matrix.shape[0] / self.fps,
            "file": npy_path,
            "label": "unassigned" # Ready for classifier
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return npy_path

    def _trigger_hardware_led(self, turn_on=True):
        """
        Sends a 1-byte command over serial to the ESP32 receiver to blink/toggle GPIO 2 LED.
        Command '1\n' turns LED on, '0\n' turns LED off.
        """
        if self.serial_port and hasattr(self.serial_port, "write"):
            try:
                cmd = b"1\n" if turn_on else b"0\n"
                self.serial_port.write(cmd)
                self.serial_port.flush()
            except Exception:
                pass

    def get_json_payload(self):
        """Returns the standard JSON-serializable status dictionary."""
        return {
            "state": self.state,
            "score": round(float(self.current_score), 4),
            "timestamp": round(time.time(), 3),
            "baseline_mean": round(float(self.baseline_mean), 4),
            "threshold_high": round(float(self.threshold_high), 4),
            "threshold_low": round(float(self.threshold_low), 4),
            "sample_count": self.gesture_sample_count
        }

    def _publish_state(self, force=False):
        """Dispatches the JSON status to registered listeners and updates state file."""
        payload = self.get_json_payload()
        self.last_publish_time = time.time()

        # Update a local JSON status file so teammates can read it anytime
        try:
            with open("latest_motion_state.json", "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

        # Call registered listener callbacks
        for listener in self.listeners:
            try:
                listener(payload)
            except Exception as e:
                print(f"[!] Listener error: {e}")
