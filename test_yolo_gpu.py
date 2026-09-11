"""
test_yolo_gpu.py
================
Verifies CUDA GPU acceleration on NVIDIA GeForce RTX 4060 with PyTorch and YOLOv8.
"""

import sys
import time
import numpy as np
import torch
from ultralytics import YOLO


def verify_gpu_and_yolo():
    print("=" * 70)
    print("        PYTORCH & YOLOV8 CUDA GPU ACCELERATION VERIFICATION")
    print("=" * 70)
    
    # 1. PyTorch CUDA Status
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch Version          : {torch.__version__}")
    print(f"CUDA Available           : {cuda_available}")
    
    if not cuda_available:
        print("\n[ERROR] CUDA is not available to PyTorch! Check CUDA driver/build.")
        sys.exit(1)
        
    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0)
    cuda_version = torch.version.cuda
    device_capability = torch.cuda.get_device_capability(0)
    
    print(f"CUDA Device Count        : {device_count}")
    print(f"PyTorch Built CUDA       : {cuda_version}")
    print(f"Detected GPU Name        : {device_name}")
    print(f"Compute Capability       : {device_capability[0]}.{device_capability[1]}")
    
    # 2. Hardware Assertion
    assert "RTX 4060" in device_name, (
        f"Assertion Failed: Expected 'RTX 4060' in device name, but got '{device_name}'"
    )
    print("Hardware Assertion       : PASSED (NVIDIA GeForce RTX 4060 verified)")
    print("-" * 70)
    
    # 3. Load YOLOv8 Model
    print("Loading YOLOv8 Nano model (yolov8n.pt)...")
    model = YOLO("yolov8n.pt")
    
    # 4. Generate Synthetic Test Frame (e.g. 640x640 BGR image)
    dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # 5. Run GPU Inference
    print("Executing GPU Inference on device=0 (NVIDIA GeForce RTX 4060)...")
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Warmup + Inference
    results = model.predict(source=dummy_frame, device=0, verbose=False)
    torch.cuda.synchronize()
    elapsed_ms = (time.time() - start_time) * 1000.0
    
    # Memory and Inference Details
    allocated_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)
    reserved_mb = torch.cuda.memory_reserved(0) / (1024 ** 2)
    
    print(f"Inference Time           : {elapsed_ms:.2f} ms")
    print(f"Inference Device Used    : {results[0].boxes.data.device}")
    print(f"Allocated GPU VRAM       : {allocated_mb:.2f} MB")
    print(f"Reserved GPU VRAM        : {reserved_mb:.2f} MB")
    print("=" * 70)
    print(">>> SUCCESS: PyTorch and YOLOv8 CUDA GPU acceleration is fully operational! <<<")
    print("=" * 70)


if __name__ == "__main__":
    verify_gpu_and_yolo()
