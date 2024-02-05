# Prepare Datasets

```
data_dir
|—— cityscapes
|   |—— leftImg8bit
|   |   |—— train
|   |   |—— val
|   |—— gtFine
|   |   |—— train
|   |   |—— val
|—— NighttimeDrivingTest
|   |—— leftImg8bit
|   |   |—— test
|   |—— gtCoarse_daytime_trainvaltest
|   |   |—— test
|—— dark_zurich
|   |—— rgb_anon
|   |   |—— val
|   |—— gt
|   |   |—— val
|—— mapillary
|   |—— training
|   |   |—— images
|   |   |—— v1.2
|   |—— validation
|   |   |—— images
|   |   |—— v1.2
|—— camvid
|   |—— images
|   |   |—— train
|   |   |—— test
|   |—— annotations
|   |   |—— train
|   |   |—— test
|—— ade
|   |—— ADEChallengeData2016
|   |   |—— images
|   |   |   |—— training
|   |   |   |—— validation
|   |   |—— annotations
|   |   |   |—— training
|   |   |   |—— validation
|—— coco_stuff164k
|   |—— images
|   |   |—— train2017
|   |   |—— val2017
|   |—— annotations
|   |   |—— train2017
|   |   |—— val2017
|—— VOCdevkit
|   |—— VOC2010
|   |   |—— JPEGImages
|   |   |—— SegmentationClassContext
|   |—— VOC2012
|   |   |—— JPEGImages
|   |   |—— SegmentationClass
|   |   |—— SegmentationClassAug
|   |   |—— SegmentationClassTrainAug
|   |—— VOCaug
|—— land
|   |—— train
|—— road
|   |—— train
|—— building
|   |—— train
|—— lits
|   |—— train
|—— kits
|   |—— train
|—— qubiq
|   |—— brain-growth
|   |—— brain-tumor
|   |—— kidney
|   |—— prostate
```

Many datasets rely on scripts from `MMSegmentation`. Please refer to [here](https://mmsegmentation.readthedocs.io/en/latest/user_guides/2_dataset_prepare.html) for more details.

## Cityscapes
* Step 1: Download the dataset from [here](https://www.cityscapes-dataset.com)
* Step 2: Run the following from `MMSegmentation`

  ```
  python tools/dataset_converters/cityscapes.py data/cityscapes
  ```

## Nighttime Driving
* Download the test set from [here](https://data.vision.ee.ethz.ch/daid/NighttimeDriving/NighttimeDrivingTest.zip).


## Dark Zurich
* Download the validation set from [here](https://data.vision.ee.ethz.ch/csakarid/shared/GCMA_UIoU/Dark_Zurich_val_anon.zip).

## Mapillary Vistas
* Step 1: Download the dataset from [here](https://www.mapillary.com/dataset/vistas)
* Step 2: Run the following
  ```
  python datas/prepare_mapillary_vistas.py path/to/data_dir
  ```

## CamVid
* Step 1: Merge the training and validation sets as the folder structure shows.

* Step 2: Run the following
  ```
  python datas/prepare_camvid.py path/to/data_dir train
  python datas/prepare_camvid.py path/to/data_dir test
  ```

## ADE20K
* Download the dataset from [here](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip)

## COCO-Stuff
* Run the following from `MMSegmentation`
  ```
  wget http://images.cocodataset.org/zips/train2017.zip
  wget http://images.cocodataset.org/zips/val2017.zip
  wget http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip

  unzip train2017.zip -d images/
  unzip val2017.zip -d images/
  unzip stuffthingmaps_trainval2017.zip -d annotations/

  python tools/dataset_converters/coco_stuff164k.py path/to/coco_stuff164k
  ```

## PASCAL VOC
* Step 1: Download PASCAL VOC 2012 from [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) and extra data from [here](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz).

* Step 2: Run the following from `MMSegmentation`
  ```
  python tools/dataset_converters/voc_aug.py \
    /path/to/data_dir/VOCdevkit \
    /path/to/data_dir/VOCdevkit/VOCaug
  ```

* Step 3: Run the following
  ```
  python datas/process_pascal_voc.py path/to/data_dir
  ```

## PASCAL Context
* Step 1: Install `detail` following [here](https://github.com/zhanghang1989/detail-api)

* Step 2: Run the following from `MMSegmentation`
  ```
  python tools/dataset_converters/pascal_context.py data/VOCdevkit data/VOCdevkit/VOC2010/trainval_merged.json
  ```

## DeepGlobe Land
* Run the following
  ```
  datas/prepare_deepglobe_land.py path/to/data_dir
  ```

## DeepGlobe Road
* Run the following
  ```
  datas/prepare_deepglobe_road.py path/to/data_dir
  ```

## DeepGlobe Building
* Run the following
  ```
  datas/prepare_deepglobe_building.py path/to/data_dir
  ```

## LiTS
* Run the following
  ```
  datas/prepare_lits_kits.py path/to/data_dir lits
  ```

## KiTS
* Run the following
  ```
  datas/prepare_lits_kits.py path/to/data_dir kits
  ```

## QUBIQ
* Run the following
  ```
  datas/prepare_qubiq.py path/to/data_dir brain-growth 0
  ```
