from pathlib import Path
from ultralytics import YOLO
import torch


# SETTINGS.


DATASET_ROOT = (
    Path.home()
    / "LunarDataset"
    / "obb_chips"
    / "split_dataset_positive_geographic_balanced_rgb"
)

DATA_YAML = "/Users/alexiswillson/LunarDataset/obb_chips/split_dataset_positive_geographic_balanced_rgb/data.yaml"
TEST_IMAGES = "/Users/alexiswillson/LunarDataset/obb_chips/split_dataset_positive_geographic_balanced_rgb/images/test"

MODEL_PATH = "/Users/alexiswillson/Documents/Git-DL-Geodynamics/DL-Geodynamics/runs/obb/runs/lunar_obb/geographic_positive_balanced_rgb/weights/best.pt"

OUTPUT_PROJECT = "runs/lunar_obb_test_predictions"
OUTPUT_NAME = "geographic_positive_balanced_rgb_test_c0.01"

IMAGE_SIZE = 640
CONF = 0.001 # Source paper uses 0.001 threshold but that's really low

SAVE_TXT = True
SAVE_CONF = True
RUN_VAL = True


# LOADING MODEL.


model = YOLO(MODEL_PATH)


# RUNNING INFERENCE ON TEST IMAGES.


print("\nRunning inference on test images...")

results = model.predict(
    source=str(TEST_IMAGES),
    imgsz=IMAGE_SIZE,
    conf=CONF,
    device=DEVICE,
    save=True,
    save_txt=SAVE_TXT,
    save_conf=SAVE_CONF,
    project=OUTPUT_PROJECT,
    name=OUTPUT_NAME,
)

print("\nInference complete.")
print(
    f"Saved prediction images and labels to: "
    f"{Path(OUTPUT_PROJECT) / OUTPUT_NAME}"
)


# FORMAL EVALUATION ON TEST SPLIT.


if RUN_VAL:
    print("\nRunning formal evaluation on test split...")

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        plots=True,
        project=OUTPUT_PROJECT,
        name=f"{OUTPUT_NAME}_val",
    )

    print("\nEvaluation complete.")
    print(metrics)