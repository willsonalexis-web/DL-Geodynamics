# DL-Geodynamics
CISC 867 Course Project at Queen's University.

# Input data to download:
A. Lunar Basemap: https://astrogeology.usgs.gov/search/map/moon_lro_lroc_wac_global_morphology_mosaic_100m
B. Wrinkle Ridge Shapefile: https://data.lroc.im-ldi.com/lroc/view_rdr/SHAPEFILE_WRINKLE_RIDGES
C. Lobate Scarp Shapefile: https://data.lroc.im-ldi.com/lroc/view_rdr/SHAPEFILE_LOBATE_SCARPS 

# Execute in order:
1. geographic_positive_dataset_curation.py
2. geographic_positive_YOLO26_full_run.py
3. geographic_positive_YOLO26_TEST_run.py

Fixed seed = 999

# Explanation of each script: 

geographic_positive_dataset_curation.py

An automated pipeline is written to curate the initial dataset for baseline reproduction. With the lunar global basemap, wrinkle ridge and lobate scarp shapefiles, the polylines are clipped to 610 by 610 pixel chips of the basemap. Only chips containing at least one polyline feature are used, to not saturate the dataset with unlabeled images; a total of 1424 geography-positive chips are exported. Polylines are converted into oriented bounding box labels, a total of 4882 labels written, 4150 wrinkle ridges and 732 lobate scarps. In addition, data is split by geographic blocks to reduce spatial leakage. The training dataset contains 69.10% of the total data, with 984 chips, 2832 wrinkle ridge labels and 545 lobate scarp labels. The validation dataset contains 14.19% of the total data, with 202 chips, 633 wrinkle ridge labels and 124 lobate scarp labels. The test dataset contains 16.71% of the total data, with 238 chips, 685 wrinkle ridges and 63 lobate scarps.

geographic_positive_YOLO26_full_run.py

Further preprocessing is applied to convert the greyscale basemap to RGB that the YOLO26 model is compatible with. YOLO26 OBB is applied for training and validation. 

geographic_positive_YOLO26_TEST_run.py

Best derived OBB model is applied to test data for final OBB detection. 
