#!/bin/bash

python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['linear_784_50', 'relu', 'linear_50_25', 'relu', 'linear_25_10']" DATASETS.CLASS_IDS "[1,5]"
python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['linear_784_50', 'relu', 'linear_50_25', 'relu', 'linear_25_10']" DATASETS.CLASS_IDS "[6,8]"
python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['linear_784_50', 'relu', 'linear_50_25', 'relu', 'linear_25_10']" DATASETS.CLASS_IDS "[2,5]"
python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['linear_784_50', 'relu', 'linear_50_25', 'relu', 'linear_25_10']" DATASETS.CLASS_IDS "[0,9]"
python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['linear_784_50', 'relu', 'linear_50_25', 'relu', 'linear_25_10']" DATASETS.CLASS_IDS "[4,7]"
python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['linear_784_50', 'relu', 'linear_50_25', 'relu', 'linear_25_10']" DATASETS.CLASS_IDS "[3,4]"
python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['linear_784_50', 'relu', 'linear_50_25', 'relu', 'linear_25_10']" DATASETS.CLASS_IDS "[0,3]"
python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['linear_784_50', 'relu', 'linear_50_25', 'relu', 'linear_25_10']" DATASETS.CLASS_IDS "[2,6]"
python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['linear_784_50', 'relu', 'linear_50_25', 'relu', 'linear_25_10']" DATASETS.CLASS_IDS "[4,8]"
python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['linear_784_50', 'relu', 'linear_50_25', 'relu', 'linear_25_10']" DATASETS.CLASS_IDS "[7,9]"



