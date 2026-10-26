#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON State Publisher for Teammate Integration (Firebase / Web / MQTT)
====================================================================
Subscribes to CSIMotionDetector and emits {state, score, timestamp} every 0.5 seconds.
Easy to plug in: just replace `handle_payload` or pass a custom callback!
"""

import json
import time

class MotionStatePublisher:
    def __init__(self, detector, custom_callback=None, verbose=True):
        """
        detector: CSIMotionDetector instance
        custom_callback: optional function(dict) to forward to Firebase/Cloud
        verbose: print JSON payload to console every 0.5s
        """
        self.detector = detector
        self.custom_callback = custom_callback
        self.verbose = verbose

        # Attach as listener to the detector
        self.detector.add_listener(self.on_detector_update)

    def on_detector_update(self, payload):
        """Called automatically every 0.5s and upon state transitions."""
        # 1. Execute teammate's custom handler (e.g. Firebase db.reference('motion').set(payload))
        if self.custom_callback:
            try:
                self.custom_callback(payload)
            except Exception as err:
                print(f"[!] Custom teammate callback failed: {err}")

        # 2. Console logging for quick verification
        if self.verbose:
            state_color = "\033[92m" if payload["state"] == "IDLE" else "\033[91m"
            reset_color = "\033[0m"
            # Format output as compact clean JSON line
            json_str = json.dumps(payload)
            # Avoid terminal escape characters in raw pipes
            print(f"[JSON PUBLISH] {json_str}")

def example_firebase_hook(payload):
    """
    Template for teammates to connect to Firebase:
    Example:
        import pyrebase / firebase_admin
        db.child("esp32_csi").child("motion").set(payload)
    """
    # Teammates can simply implement their Firebase / MQTT push logic here
    pass
