'''
Lunar Dataset Curation and Splitting Script.

Consult README for URLs for input download. 

An automated pipeline is written to curate the initial dataset for baseline reproduction. With the lunar global 
basemap, wrinkle ridge and lobate scarp shapefiles, the polylines are clipped to 610 by 610 pixel chips of the 
basemap. Only chips containing at least one polyline feature are used, to not saturate the dataset with unlabeled 
images; a total of 1424 geography-positive chips are exported. Polylines are converted into oriented bounding box 
labels, a total of 4882 labels written, 4150 wrinkle ridges and 732 lobate scarps. In addition, data is split by 
geographic blocks to reduce spatial leakage. The training dataset contains 69.10% of the total data, with 984 chips, 
2832 wrinkle ridge labels and 545 lobate scarp labels. The validation dataset contains 14.19% of the total data, with 
202 chips, 633 wrinkle ridge labels and 124 lobate scarp labels. The test dataset contains 16.71% of the total data, 
with 238 chips, 685 wrinkle ridges and 63 lobate scarps.

Alexis Willson
October 2026
Queen's Univeristy
'''

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
