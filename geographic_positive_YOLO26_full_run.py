from pathlib import Path
from shutil import copy2

from PIL import Image
from ultralytics import YOLO
import torch


# SETTING DATASET AND MODEL.


ORIG_DATASET_ROOT = (
    Path.home()
    / "LunarDataset"
    / "obb_chips"
    / "split_dataset_positive_geographic_balanced"
)

RGB_DATASET_ROOT = (
    Path.home()
    / "LunarDataset"
    / "obb_chips"
    / "split_dataset_positive_geographic_balanced_rgb"
)

ORIG_DATA_YAML = ORIG_DATASET_ROOT / "data.yaml"
RGB_DATA_YAML = RGB_DATASET_ROOT / "data.yaml"

MODEL_NAME = "yolo26n-obb.pt"


# TRAINING PARAMETERS.


IMAGE_SIZE = 640
EPOCHS = 50
BATCH_SIZE = 4
WORKERS = 1

if torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

SEED = 999
DETERMINISTIC = False

TRAIN_PROJECT = "runs/lunar_obb"
TRAIN_NAME = "geographic_positive_balanced_rgb_seed999"

CONVERT_IMAGES_TO_RGB = True
OVERWRITE_RGB_IMAGES = False   # Set True to recreate RGB files


# CHECKING ORIGINAL DATASET.


if not ORIG_DATA_YAML.exists():
    raise FileNotFoundError(
        f"Original dataset configuration not found: {ORIG_DATA_YAML}"
    )


# CREATING RGB DATASET COPY.


def convert_dataset_to_rgb(src_root: Path, dst_root: Path, overwrite: bool = False):
    """
    Convert all dataset images to RGB and copy labels/YAML structure.
    """
    for split in ["train", "val", "test"]:
        src_img_dir = src_root / "images" / split
        dst_img_dir = dst_root / "images" / split
        src_lbl_dir = src_root / "labels" / split
        dst_lbl_dir = dst_root / "labels" / split

        if not src_img_dir.exists():
            raise FileNotFoundError(f"Missing image split folder: {src_img_dir}")
        if not src_lbl_dir.exists():
            raise FileNotFoundError(f"Missing label split folder: {src_lbl_dir}")

        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        # Convert images to RGB
        for img_path in src_img_dir.glob("*.*"):
            if img_path.name == ".DS_Store":
                continue

            dst_img_path = dst_img_dir / img_path.name

            if dst_img_path.exists() and not overwrite:
                continue

            with Image.open(img_path) as img:
                rgb_img = img.convert("RGB")
                rgb_img.save(dst_img_path)

        # Copy label files unchanged
        for lbl_path in src_lbl_dir.glob("*.txt"):
            dst_lbl_path = dst_lbl_dir / lbl_path.name

            if dst_lbl_path.exists() and not overwrite:
                continue

            copy2(lbl_path, dst_lbl_path)

    # Writing new data.yaml for RGB dataset
    yaml_text = f"""path: {dst_root}
train: images/train
val: images/val
test: images/test

names:
  0: wrinkle_ridge
  1: lobate_scarp
"""
    dst_yaml = dst_root / "data.yaml"
    dst_yaml.write_text(yaml_text, encoding="utf-8")
    return dst_yaml


if CONVERT_IMAGES_TO_RGB:
    print("Converting dataset images to RGB...")
    RGB_DATA_YAML = convert_dataset_to_rgb(
        src_root=ORIG_DATASET_ROOT,
        dst_root=RGB_DATASET_ROOT,
        overwrite=OVERWRITE_RGB_IMAGES,
    )
    print(f"RGB dataset ready at: {RGB_DATASET_ROOT}")
else:
    if not RGB_DATA_YAML.exists():
        raise FileNotFoundError(
            f"RGB dataset YAML not found: {RGB_DATA_YAML}"
        )



# LOADING MODEL.


print(f"Using device: {DEVICE}")
print(f"Training dataset: {RGB_DATA_YAML}")

model = YOLO(MODEL_NAME)


# TRAINING MODEL ON RGB DATASET.


results = model.train(
    data=str(RGB_DATA_YAML),
    imgsz=IMAGE_SIZE,
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    workers=WORKERS,
    device=DEVICE,
    seed=SEED,
    deterministic=DETERMINISTIC,
    project=TRAIN_PROJECT,
    name=TRAIN_NAME,
)


# REPORTING RESULTS LOCATION.


print("\nTraining finished.")
print(f"Results saved to: {results.save_dir}")
print(f"Trained on RGB dataset: {RGB_DATA_YAML}")