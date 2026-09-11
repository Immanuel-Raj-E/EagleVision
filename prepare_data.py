"""Dataset Structure & Configuration Auto-Discovery Script for Disaster Survivor Dataset."""
import os
import glob
from pathlib import Path
import yaml

DATASET_DIR = Path(r"d:\SEC\dataset").resolve()
if not DATASET_DIR.exists():
    DATASET_DIR = Path(r"d:\SEC\data").resolve()

print("=" * 65)
print("SAR SURVIVOR DATASET AUTO-DISCOVERY & VALIDATION")
print("=" * 65)
print(f"Inspecting dataset at: {DATASET_DIR}")

# 1. Discover split folders
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

# Possible directory patterns
# Pattern A: images/train, images/val, labels/train, labels/val
# Pattern B: train/images, val/images, train/labels, val/labels
splits_found = {}

for split in ['train', 'val', 'valid', 'test']:
    split_norm = 'val' if split == 'valid' else split
    
    # Check Pattern A: images/<split> and labels/<split>
    img_dir_a = DATASET_DIR / "images" / split
    lbl_dir_a = DATASET_DIR / "labels" / split
    
    # Check Pattern B: <split>/images and <split>/labels
    img_dir_b = DATASET_DIR / split / "images"
    lbl_dir_b = DATASET_DIR / split / "labels"

    if img_dir_a.exists():
        img_dir = img_dir_a
        lbl_dir = lbl_dir_a
        rel_img = f"images/{split}"
    elif img_dir_b.exists():
        img_dir = img_dir_b
        lbl_dir = lbl_dir_b
        rel_img = f"{split}/images"
    else:
        continue

    # Count valid files
    images = [f for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS]
    labels = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

    splits_found[split_norm] = {
        'img_dir': img_dir,
        'lbl_dir': lbl_dir,
        'rel_img': rel_img,
        'num_images': len(images),
        'num_labels': len(labels)
    }

print("\n--- Split Discovery & Annotation Counts ---")
for split_name, info in splits_found.items():
    print(f"[{split_name.upper()}] Images: {info['num_images']:,} ({info['img_dir']})")
    print(f"        Labels: {info['num_labels']:,} ({info['lbl_dir']})")

if 'train' not in splits_found or splits_found['train']['num_images'] == 0:
    raise RuntimeError(f"No training images discovered in {DATASET_DIR}!")

# 2. Inspect label classes
class_counts = {}
sample_labels = list(splits_found['train']['lbl_dir'].glob("*.txt"))[:1000]
for lbl_file in sample_labels:
    with open(lbl_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                try:
                    c_id = int(parts[0])
                    class_counts[c_id] = class_counts.get(c_id, 0) + 1
                except ValueError:
                    pass

print("\n--- Class ID Distribution (Sample of 1,000 files) ---")
for c_id, count in sorted(class_counts.items()):
    print(f"  Class {c_id}: {count:,} instances")

# 3. Generate data.yaml
yaml_path = DATASET_DIR / "data.yaml"

train_rel = splits_found['train']['rel_img']
val_rel = splits_found.get('val', splits_found['train'])['rel_img']
test_rel = splits_found['test']['rel_img'] if 'test' in splits_found else None

data_config = {
    'path': str(DATASET_DIR).replace('\\', '/'),
    'train': train_rel.replace('\\', '/'),
    'val': val_rel.replace('\\', '/'),
    'nc': 2,
    'names': {
        0: 'human',
        1: 'animal'
    }
}
if test_rel:
    data_config['test'] = test_rel.replace('\\', '/')

with open(yaml_path, 'w', encoding='utf-8') as f:
    yaml.dump(data_config, f, sort_keys=False, default_flow_style=False)

print(f"\nGenerated normalized data.yaml at: {yaml_path}")
print("data.yaml content:")
with open(yaml_path, 'r', encoding='utf-8') as f:
    print(f.read().strip())
print("=" * 65)
print("DATASET VALIDATION COMPLETE & READY FOR TRAINING")
print("=" * 65)
