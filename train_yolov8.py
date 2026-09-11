"""Fine-Tuning Execution Script for SAR Disaster Survivor Detection on YOLOv8s."""
import os
import sys
from pathlib import Path
import torch

def run_training():
    print("=" * 65)
    print("SAR YOLOv8s SURVIVOR FINE-TUNING PIPELINE")
    print("=" * 65)

    # 1. Hardware & CUDA Verification
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! Training requires an NVIDIA GPU (RTX 4060).")
    
    device_name = torch.cuda.get_device_name(0)
    device_cap = torch.cuda.get_device_capability(0)
    print(f"CUDA Device Detected: {device_name} (Capability {device_cap})")
    print(f"Allocated VRAM: {torch.cuda.memory_allocated(0)/(1024**2):.1f} MB | Free VRAM: {torch.cuda.mem_get_info()[0]/(1024**2):.1f} MB")

    # 2. Dataset Verification
    data_yaml_path = Path(r"d:\SEC\dataset\data.yaml")
    if not data_yaml_path.exists():
        data_yaml_path = Path(r"d:\SEC\data\data.yaml")
    
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}. Run prepare_data.py first.")

    # Check images count
    dataset_dir = data_yaml_path.parent
    train_images = list((dataset_dir / "images" / "train").glob("*.*"))
    print(f"Verified {len(train_images)} training images in dataset.")
    if len(train_images) < 10:
        raise ValueError(f"CRITICAL ERROR: Insufficient training images found ({len(train_images)} < 10). Halting.")

    # 3. Model Weights Initialization
    from ultralytics import YOLO
    
    weights_path = "yolov8s.pt"
    print(f"\nLoading base weights: {weights_path} (YOLOv8-Small for high small-target recall)...")
    model = YOLO(weights_path)
    
    print("\nStarting fine-tuning with SAR drone hyperparameters...")
    print("Freezing early backbone feature extractor (freeze=10: layers 0-9 locked, training neck & heads)...")

    # 4. Training Execution
    results = model.train(
        data=str(data_yaml_path).replace('\\', '/'),
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        amp=True,
        freeze=10,
        lr0=0.001,
        lrf=0.01,
        mosaic=1.0,
        mixup=0.15,
        degrees=15.0,
        flipud=0.3,
        fliplr=0.5,
        project="d:/SEC/runs/sar_finetune",
        name="yolov8s_sar",
        save=True,
        exist_ok=True,
        workers=4,
        verbose=True
    )

    best_pt = Path(r"d:\SEC\runs\sar_finetune\yolov8s_sar\weights\best.pt")
    print("\n" + "=" * 65)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 65)
    print(f"Best model weights saved to: {best_pt}")
    if best_pt.exists():
        print(f"Model File Size: {best_pt.stat().st_size / (1024**2):.2f} MB")
    
    return results

if __name__ == "__main__":
    run_training()
