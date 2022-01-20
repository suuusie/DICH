# Deep Incremental Cross-modal Hashing (DICH)
We propose a novel Deep Incremental Cross-modal Hashing method, which could directly learn incremental hash codes for newly coming data with unseen concepts while keeping original hash codes for old data unchanged under the pursuit of fast training up-to-data model.

## REQUIREMENTS
1. pytorch>=1.0
2. loguru

## DATASETS
We use two benchmark datasets to evaluate the effectiveness of our method. You can download the MIRFlickr and NUS-WIDE by the following links:
1. [MIRFlickr](https://pan.baidu.com/s/1WHBWnB2fYuvQM5Oh9SETzA) Password: mmc2
2. [NUS-WIDE](https://pan.baidu.com/s/1Q06mT-hi6K_yMEyTz94TSg) Password: nsdb

## Pre-trained CNN-F model
You can download the CNN-F model pre-trained on ImageNet by the following links:
[Pre-trained CNN-F model](https://pan.baidu.com/s/1q79n5mPqEnVazTTSfxQ20g) Password: 42v4

## USAGE
A demo for running DICH on MIRFlickr:
1. Put the MIRFlickr dataset to ./dataset/MIRFlickr.
2. Put the original hash codes learned from existing models to ./checkpoints. (Here, we provide original hash codes generated from DCMH on MIRFlickr dataset with the split of 3 unseen categories in the case of 16 bits.)
3. Put the pre-trained CNN-F model to ./models
4. Run run_incremental_dcmh1.py

PS. You can easily run DICH on NUS-WIDE by using NUS-WIDE dataset and change the configuration in run_incremental.py. It should be noticed that before you run the run_incremental_dcmh1.py, you should get the corresponding learned original hash codes.



