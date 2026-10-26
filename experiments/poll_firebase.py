#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import json
import urllib.request
import urllib.error

URL = "https://espcsi-default-rtdb.asia-southeast1.firebasedatabase.app/room_os/live_state.json"
TOTAL_DURATION = 120  # 2 minutes
INTERVAL = 5          # 5 seconds
NUM_POLLS = TOTAL_DURATION // INTERVAL  # 24 polls

def fetch_data():
    try:
        req = urllib.request.Request(URL, headers={"User-Agent": "CSI-Monitor/1.0"})
        with urllib.request.urlopen(req, timeout=4) as response:
            data = json.loads(response.read().decode('utf-8'))
            return data
    except Exception as e:
        return {"error": str(e)}

def main():
    print("=" * 90)
    print(f"POLLING FIREBASE LIVE_STATE ({NUM_POLLS} POLLS @ 5s INTERVAL = 2 MINUTES)")
    print("=" * 90)
    header = f"{'#':<3} | {'iso_time':<22} | {'activity':<16} | {'motion_level':<12} | {'baseline_dev':<14} | {'pkt_rate_hz':<12}"
    print(header)
    print("-" * len(header))

    records = []
    iso_times = []

    start_time = time.time()

    for i in range(1, NUM_POLLS + 1):
        t0 = time.time()
        data = fetch_data()

        if "error" in data:
            iso = "ERROR"
            act = data["error"][:15]
            mot = "N/A"
            dev = "N/A"
            rate = "N/A"
        else:
            iso = str(data.get("iso_time", "N/A"))
            act = str(data.get("activity", "N/A"))
            mot = f"{data.get('motion_level', 0.0):.2f}" if isinstance(data.get('motion_level'), (int, float)) else str(data.get('motion_level'))
            dev = f"{data.get('baseline_deviation', 0.0):.2f}" if isinstance(data.get('baseline_deviation'), (int, float)) else str(data.get('baseline_deviation'))
            rate = f"{data.get('esp32_packet_rate_hz', 0.0):.1f}" if isinstance(data.get('esp32_packet_rate_hz'), (int, float)) else str(data.get('esp32_packet_rate_hz'))
            iso_times.append(iso)

        row_str = f"{i:<3} | {iso:<22} | {act:<16} | {mot:<12} | {dev:<14} | {rate:<12}"
        print(row_str, flush=True)

        records.append({
            "idx": i,
            "iso_time": iso,
            "activity": act,
            "motion_level": mot,
            "baseline_deviation": dev,
            "esp32_packet_rate_hz": rate
        })

        if i < NUM_POLLS:
            elapsed = time.time() - t0
            sleep_time = max(0.0, INTERVAL - elapsed)
            time.sleep(sleep_time)

    print("-" * len(header))
    print(f"\nCompleted {NUM_POLLS} polls in {time.time() - start_time:.1f}s.")

    unique_iso = set(iso_times)
    print(f"\nUnique iso_time values observed: {len(unique_iso)}")
    for u in sorted(unique_iso):
        print(f"  -> {u}")

    if len(unique_iso) > 1:
        print("\n[RESULT] iso_time CHANGED during the 2-minute polling period.")
    else:
        print("\n[RESULT] iso_time DID NOT CHANGE during the 2-minute polling period (STAGNANT).")

    # Save summary json
    with open("firebase_poll_results.json", "w", encoding="utf-8") as f:
        json.dump({"records": records, "unique_iso_times": list(unique_iso)}, f, indent=2)

if __name__ == "__main__":
    main()
