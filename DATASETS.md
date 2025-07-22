# Benchmark Datasets

We evaluated Llama3-MS-CLIP on several Earth Observation datasets, using their test splits.

All benchmark datasets besides BigEarthNet_RGB, BigEarthNet_MS and ForestNet_RGB have to be organized in an ImageFolder format with the following folder structure:

```
  benchmark_datasets/dataset_name/class_1/xxx.png
  benchmark_datasets/dataset_name/class_1/xxy.png
  benchmark_datasets/dataset_name/class_1/xxz.png
  .
  .
  .
  benchmark_datasets/dataset_name/class_x/xxx.tif
  benchmark_datasets/dataset_name/class_x/xxy.tif
  benchmark_datasets/dataset_name/class_x/xxz.tif
```

The dataset_name from the above folder structure should match the following names:

```
METERML_RGB,
EuroSAT_RGB,
RESISC45_RGB,
AID_RGB,
METERML_NAIP,
METERML_MS,
EuroSAT_MS,
```

Both BigEarthNet_RGB and BigEarthNet_MS use a single dataset called BigEarthNet and depending on whether the mode is RGB or MS the appropriate bands will be selected automatically. The dataset is organized in the following format:
```
images are composed of multiple single channel geotiffs
labels are multiclass, stored in a single json file per image
```

The following is the folder structure for BigEarthNet: 

```
benchmark_datasets/BigEarthNet/sentinel-2/xxx_image_folder/band_1.tif
.
.
benchmark_datasets/BigEarthNet/sentinel-2/xxx_image_folder/band_12.tif
benchmark_datasets/BigEarthNet/sentinel-2/xxx_image_folder_name/labels_metadata.json
.
.
.
.
benchmark_datasets/BigEarthNet/sentinel-2/xxz_image_folder/band_12.tif
benchmark_datasets/BigEarthNet/sentinel-2/xxz_image_folder_name/labels_metadata.json
```

ForestNet_RGB contains 3 csv files for train test and validation and these csv files contain the image paths and the class labels. Benchmarking is done on the test split. The following is the folder structure for ForestNet_RGB: 

```
benchmark_datasets/ForestNet_RGB/examples/xxx_image_folder/images/visible/composite.png
.
.
benchmark_datasets/ForestNet_RGB/examples/xxz_image_folder/images/visible/composite.png
.
.
.
.
benchmark_datasets/ForestNet_RGB/test_csv
benchmark_datasets/ForestNet_RGB/val_csv
benchmark_datasets/ForestNet_RGB/train_csv
```