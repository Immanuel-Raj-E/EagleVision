"""
download_weights.py
===================
Utility script to ensure all necessary model weights exist locally.
Downloads pre-trained thermal and baseline YOLOv8 weights if not present.
"""

import os
import urllib.request
import sys

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

THERMAL_URL = "https://huggingface.co/KissTheHabit/yolov8n-hituav-thermal-finetune/resolve/main/yolov8n_hituav_thermal.pt"
THERMAL_PATH = os.path.join(WEIGHTS_DIR, "thermal_weights.pt")

def download_file(url: str, dest_path: str):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"[✓] Weights already exist at: {dest_path}")
        return
    print(f"[↓] Downloading model weights from {url}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"[✓] Successfully downloaded to {dest_path} ({os.path.getsize(dest_path):,} bytes)")
    except Exception as e:
        print(f"[!] Warning: Could not download weights automatically ({e}).")

if __name__ == "__main__":
    download_file(THERMAL_URL, THERMAL_PATH)
