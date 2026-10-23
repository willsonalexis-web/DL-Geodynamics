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
import csv
import random

import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt

from PIL import Image
from rasterio.windows import Window
from shapely.geometry import box
from shapely.ops import transform
from matplotlib.patches import Rectangle


# SETTINGS. 
# Input path to basemap, wrinkle shapefile, scarp shapefile 

BASEMAP = Path("/Users/alexiswillson/LunarDataset/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif")
WRINKLE_SHP = Path("/Users/alexiswillson/LunarDataset/WRINKLE_RIDGES_180/WRINKLE_RIDGES_180.shp")
SCARP_SHP = Path("/Users/alexiswillson/LunarDataset/LOBATE_SCARPS_180/LOBATE_SCARPS_180.shp")

# Creating new output folder
OUT = Path.home() / "LunarDataset" / "obb_chips" / "split_dataset_positive_geographic_balanced"

CHIP_SIZE = 610
LINE_WIDTH_PX = 4.0

WRINKLE_CLASS = 0
SCARP_CLASS = 1
CLASS_NAMES = {
    0: "wrinkle_ridge",
    1: "lobate_scarp",
}

MIN_LINE_FRACTION = 0.20

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

# Each geographic block spans 10 x 10 chips
BLOCK_SIZE_CHIPS = 10

# Class-balance weighting for block assignment. Higher value pushes assignment to better balance that class.
MINORITY_CLASS_EXTRA_WEIGHT = 2.0

# Plot output
MAP_PATH = OUT / "geographic_split_overview.png"

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9

IMAGE_DIR = OUT / "images"
LABEL_DIR = OUT / "labels"

for split in ["train", "val", "test"]:
    (IMAGE_DIR / split).mkdir(parents=True, exist_ok=True)
    (LABEL_DIR / split).mkdir(parents=True, exist_ok=True)

OUT.mkdir(parents=True, exist_ok=True)


# HELP FUNCTIONS. 

def to_pixel_geometry(geom, inverse_transform):
    """Convert map-coordinate geometry to raster pixel coordinates."""
    def convert_xy(x, y, z=None):
        col = inverse_transform.a * x + inverse_transform.b * y + inverse_transform.c
        row = inverse_transform.d * x + inverse_transform.e * y + inverse_transform.f
        return (col, row) if z is None else (col, row, z)

    return transform(convert_xy, geom)


def line_parts(geom):
    """Return individual line parts from line or multiline geometry."""
    if geom is None or geom.is_empty:
        return []

    if geom.geom_type == "LineString":
        return [geom]

    if geom.geom_type == "MultiLineString":
        return list(geom.geoms)

    if geom.geom_type == "GeometryCollection":
        parts = []
        for part in geom.geoms:
            parts.extend(line_parts(part))
        return parts

    return []


def save_png(data, path):
    """Save a raster chip as an 8-bit PNG."""
    if data.shape[0] == 1:
        arr = data[0].astype(np.float32)
        valid = np.isfinite(arr)

        if not valid.any():
            arr = np.zeros(arr.shape, dtype=np.uint8)
        else:
            lo, hi = np.percentile(arr[valid], [2, 98])
            if hi <= lo:
                hi = lo + 1
            arr = np.clip((arr - lo) / (hi - lo), 0, 1)
            arr[~valid] = 0
            arr = (arr * 255).astype(np.uint8)

        Image.fromarray(arr).save(path)

    else:
        arr = np.moveaxis(data[:3], 0, -1).astype(np.float32)
        rgb = np.zeros(arr.shape, dtype=np.uint8)

        for b in range(3):
            band = arr[:, :, b]
            valid = np.isfinite(band)

            if valid.any():
                lo, hi = np.percentile(band[valid], [2, 98])
                if hi <= lo:
                    hi = lo + 1

                scaled = np.clip((band - lo) / (hi - lo), 0, 1)
                scaled[~valid] = 0
                rgb[:, :, b] = (scaled * 255).astype(np.uint8)

        Image.fromarray(rgb).save(path)


def make_obb(line, tile_size, line_width):
    """
    Make an OBB around a clipped line in LOCAL CHIP PIXELS.
    Returns four corners, or None if invalid.
    """
    strip = line.buffer(line_width / 2, cap_style="flat")

    if strip.is_empty:
        return None

    rect = strip.minimum_rotated_rectangle

    if rect.geom_type != "Polygon":
        return None

    coords = list(rect.exterior.coords)[:-1]

    if len(coords) != 4:
        return None

    eps = 1e-6
    for x, y in coords:
        if (
            x < -eps or y < -eps
            or x > tile_size + eps
            or y > tile_size + eps
        ):
            return None

    normalized = []
    for x, y in coords:
        x = min(max(x, 0.0), float(tile_size))
        y = min(max(y, 0.0), float(tile_size))
        normalized.extend([x / tile_size, y / tile_size])

    return normalized


# LOADING FEATURES IN PIXEL SPACE.


with rasterio.open(BASEMAP) as src:

    if src.crs is None:
        raise ValueError(
            "The basemap has no CRS. Establish its correct "
            "coordinate reference system before continuing."
        )

    raster_crs = src.crs
    inverse_transform = ~src.transform

    sources = [
        (WRINKLE_SHP, WRINKLE_CLASS, "wrinkle_ridge"),
        (SCARP_SHP, SCARP_CLASS, "lobate_scarp"),
    ]

    features = []

    for shp_path, class_id, class_name in sources:
        gdf = gpd.read_file(shp_path)

        if gdf.crs is None:
            raise ValueError(
                f"{shp_path.name} has no CRS. Define its actual "
                "CRS before running this script."
            )

        gdf = gdf.to_crs(raster_crs)

        for row_id, row in gdf.iterrows():
            geom = row.geometry

            if geom is None or geom.is_empty:
                continue

            source_id = str(row.get("id", row_id))
            source_length = float(geom.length)

            pix_geom = to_pixel_geometry(geom, inverse_transform)

            for part_id, line in enumerate(line_parts(pix_geom)):
                if line.length > 0:
                    features.append({
                        "class_id": class_id,
                        "class_name": class_name,
                        "source_file": shp_path.name,
                        "source_id": source_id,
                        "part_id": part_id,
                        "source_length_crs_units": source_length,
                        "geometry": line,
                    })

    print(f"Loaded {len(features)} line parts.")

    
    # BUILDING ALL POSITIVE CHIP RECORDS FIRST.
    

    chip_records = []
    skipped_boxes = 0
    total_valid_boxes = 0

    for row_off in range(0, src.height, CHIP_SIZE):
        for col_off in range(0, src.width, CHIP_SIZE):

            if (
                row_off + CHIP_SIZE > src.height
                or col_off + CHIP_SIZE > src.width
            ):
                continue

            tile = box(
                col_off, row_off,
                col_off + CHIP_SIZE, row_off + CHIP_SIZE
            )

            candidates = [
                f for f in features
                if f["geometry"].intersects(tile)
            ]

            if not candidates:
                continue

            label_lines = []
            chip_metadata_rows = []
            class_label_counts = {WRINKLE_CLASS: 0, SCARP_CLASS: 0}

            for feature in candidates:
                line = feature["geometry"]

                clipped = line.intersection(tile)
                parts = line_parts(clipped)

                for clipped_part_id, part in enumerate(parts):
                    if part.length <= 0:
                        continue

                    fraction = part.length / line.length

                    if fraction < MIN_LINE_FRACTION:
                        skipped_boxes += 1
                        continue

                    local_line = to_pixel_geometry(
                        part,
                        rasterio.Affine.translation(-col_off, -row_off)
                    )

                    corners = make_obb(local_line, CHIP_SIZE, LINE_WIDTH_PX)

                    if corners is None:
                        skipped_boxes += 1
                        continue

                    coords_text = " ".join(f"{v:.6f}" for v in corners)
                    label_lines.append(f"{feature['class_id']} {coords_text}")
                    total_valid_boxes += 1
                    class_label_counts[feature["class_id"]] += 1

                    chip_metadata_rows.append({
                        "class_id": feature["class_id"],
                        "class_name": feature["class_name"],
                        "source_file": feature["source_file"],
                        "source_id": feature["source_id"],
                        "part_id": feature["part_id"],
                        "clipped_part_id": clipped_part_id,
                        "source_length_crs_units": feature["source_length_crs_units"],
                        "line_length_px_in_chip": part.length,
                        "line_fraction_in_chip": fraction,
                        "col_offset": col_off,
                        "row_offset": row_off,
                        "line_width_px": LINE_WIDTH_PX,
                    })

            if not label_lines:
                continue

            block_x = int(col_off // (BLOCK_SIZE_CHIPS * CHIP_SIZE))
            block_y = int(row_off // (BLOCK_SIZE_CHIPS * CHIP_SIZE))

            chip_records.append({
                "row_off": row_off,
                "col_off": col_off,
                "block_x": block_x,
                "block_y": block_y,
                "label_lines": label_lines,
                "metadata_rows": chip_metadata_rows,
                "n_labels": len(label_lines),
                "class_label_counts": class_label_counts,
            })

    print(f"Positive chips with valid labels: {len(chip_records)}")

    if not chip_records:
        raise RuntimeError("No positive chips were generated.")

    
    # SUMMARIZING CLASS TOTALS AND IDENTIFYING MINORITY CLASS.
    

    total_chip_count = len(chip_records)
    total_class_counts = {WRINKLE_CLASS: 0, SCARP_CLASS: 0}

    for chip in chip_records:
        for class_id, count in chip["class_label_counts"].items():
            total_class_counts[class_id] += count

    minority_class = min(total_class_counts, key=total_class_counts.get)

    print("Total label counts:")
    for class_id, count in total_class_counts.items():
        print(f"  {CLASS_NAMES[class_id]}: {count}")
    print(f"Minority class: {CLASS_NAMES[minority_class]}")

    
    # SUMMARIZING BY GEOGRAPHIC BLOCK.
    

    block_stats = {}

    for chip in chip_records:
        key = (chip["block_x"], chip["block_y"])

        if key not in block_stats:
            block_stats[key] = {
                "block_x": chip["block_x"],
                "block_y": chip["block_y"],
                "n_chips": 0,
                "class_counts": {WRINKLE_CLASS: 0, SCARP_CLASS: 0},
            }

        block_stats[key]["n_chips"] += 1

        for class_id, count in chip["class_label_counts"].items():
            block_stats[key]["class_counts"][class_id] += count

    blocks = list(block_stats.values())

    random.seed(SEED)
    random.shuffle(blocks)

    
    # GEOGRAPHIC BLOCK SPLITTING WITH CLASS-BALANCE HEURISTIC.
    

    split_names = ["train", "val", "test"]
    split_ratios = {
        "train": TRAIN_RATIO,
        "val": VAL_RATIO,
        "test": TEST_RATIO,
    }

    target_chip_counts = {
        split: ratio * total_chip_count
        for split, ratio in split_ratios.items()
    }

    target_class_counts = {
        split: {
            class_id: ratio * total_class_counts[class_id]
            for class_id in total_class_counts
        }
        for split, ratio in split_ratios.items()
    }

    current_chip_counts = {split: 0 for split in split_names}
    current_class_counts = {
        split: {WRINKLE_CLASS: 0, SCARP_CLASS: 0}
        for split in split_names
    }

    block_assignment = {}

    for block in blocks:
        best_split = None
        best_score = None

        for split in split_names:
            chip_deficit = (
                target_chip_counts[split] - current_chip_counts[split]
            )

            score = chip_deficit

            for class_id in [WRINKLE_CLASS, SCARP_CLASS]:
                class_deficit = (
                    target_class_counts[split][class_id]
                    - current_class_counts[split][class_id]
                )

                weight = 1.0
                if class_id == minority_class:
                    weight = MINORITY_CLASS_EXTRA_WEIGHT

                score += weight * class_deficit

            if best_score is None or score > best_score:
                best_score = score
                best_split = split

        key = (block["block_x"], block["block_y"])
        block_assignment[key] = best_split

        current_chip_counts[best_split] += block["n_chips"]

        for class_id in [WRINKLE_CLASS, SCARP_CLASS]:
            current_class_counts[best_split][class_id] += block["class_counts"][class_id]

    for chip in chip_records:
        key = (chip["block_x"], chip["block_y"])
        chip["split"] = block_assignment[key]

    
    # EXPORTING CHIPS TO SPLIT FOLDERS.
    

    metadata_rows = []
    split_manifest_rows = []

    for chip_index, chip in enumerate(chip_records):
        split = chip["split"]
        row_off = chip["row_off"]
        col_off = chip["col_off"]

        window = Window(col_off, row_off, CHIP_SIZE, CHIP_SIZE)
        data = src.read(window=window)

        if data.shape[1:] != (CHIP_SIZE, CHIP_SIZE):
            continue

        chip_name = f"chip_{chip_index:06d}"

        image_path = IMAGE_DIR / split / f"{chip_name}.png"
        label_path = LABEL_DIR / split / f"{chip_name}.txt"

        save_png(data, image_path)
        label_path.write_text("\n".join(chip["label_lines"]), encoding="utf-8")

        split_manifest_rows.append({
            "chip": chip_name,
            "split": split,
            "row_offset": row_off,
            "col_offset": col_off,
            "block_x": chip["block_x"],
            "block_y": chip["block_y"],
            "n_labels": chip["n_labels"],
            "wrinkle_ridge_labels": chip["class_label_counts"][WRINKLE_CLASS],
            "lobate_scarp_labels": chip["class_label_counts"][SCARP_CLASS],
        })

        for row in chip["metadata_rows"]:
            row["chip"] = chip_name
            row["split"] = split
            metadata_rows.append(row)

    
    # WRITING DATA TABLES.
    

    data_yaml = OUT / "data.yaml"
    data_yaml.write_text(
        f"""path: {OUT}
train: images/train
val: images/val
test: images/test

names:
  0: wrinkle_ridge
  1: lobate_scarp
""",
        encoding="utf-8"
    )

    metadata_path = OUT / "feature_metadata.csv"
    if metadata_rows:
        with metadata_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=metadata_rows[0].keys())
            writer.writeheader()
            writer.writerows(metadata_rows)

    manifest_path = OUT / "split_manifest.csv"
    if split_manifest_rows:
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=split_manifest_rows[0].keys())
            writer.writeheader()
            writer.writerows(split_manifest_rows)

    # Split-wise class counts
    manifest_df = pd.DataFrame(split_manifest_rows)

    class_count_rows = []
    for split in split_names:
        subset = manifest_df[manifest_df["split"] == split]
        class_count_rows.append({
            "split": split,
            "n_chips": len(subset),
            "wrinkle_ridge_labels": int(subset["wrinkle_ridge_labels"].sum()),
            "lobate_scarp_labels": int(subset["lobate_scarp_labels"].sum()),
            "total_labels": int(subset["wrinkle_ridge_labels"].sum() + subset["lobate_scarp_labels"].sum()),
        })

    class_counts_df = pd.DataFrame(class_count_rows)
    class_counts_path = OUT / "class_counts_by_split.csv"
    class_counts_df.to_csv(class_counts_path, index=False)


    # REPORT STATS.
    

    print("\nFinished.")
    print(f"Positive chips exported: {len(split_manifest_rows)}")
    print(f"OBB labels written: {total_valid_boxes}")
    print(f"OBB labels skipped: {skipped_boxes}")
    print(f"Spatial blocks: {len(blocks)}")

    for split in split_names:
        chip_count = int((manifest_df["split"] == split).sum())
        percentage = 100 * chip_count / len(manifest_df)

        wr = int(
            manifest_df.loc[manifest_df["split"] == split, "wrinkle_ridge_labels"].sum()
        )
        ls = int(
            manifest_df.loc[manifest_df["split"] == split, "lobate_scarp_labels"].sum()
        )

        print(f"{split}: {chip_count:,} chips ({percentage:.2f}%) | "
              f"wrinkle_ridge={wr} | lobate_scarp={ls}")

    print(f"\nOutput folder: {OUT}")
    print(f"Metadata: {metadata_path}")
    print(f"Split manifest: {manifest_path}")
    print(f"Class counts: {class_counts_path}")
    print(f"Split map: {MAP_PATH}")
    print(f"Data YAML: {data_yaml}")
