#!/bin/bash

#python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['linear_784_50', 'relu', 'linear_50_25', 'relu', 'linear_25_25', 'relu', 'linear_25_10']" DATASETS.CLASS_IDS "[0,1]" MODEL.TYPE "MLP4"
#python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['linear_784_50', 'relu', 'linear_50_25', 'relu', 'linear_25_25', 'relu', 'linear_25_10']" DATASETS.CLASS_IDS "[1,4]" MODEL.TYPE "MLP4"
#python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['linear_784_50', 'relu', 'linear_50_25', 'relu', 'linear_25_25', 'relu', 'linear_25_10']" DATASETS.CLASS_IDS "[2,7]" MODEL.TYPE "MLP4"
#python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['linear_784_50', 'relu', 'linear_50_25', 'relu', 'linear_25_25', 'relu', 'linear_25_10']" DATASETS.CLASS_IDS "[4,9]" MODEL.TYPE "MLP4"
#python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['linear_784_50', 'relu', 'linear_50_25', 'relu', 'linear_25_25', 'relu', 'linear_25_10']" DATASETS.CLASS_IDS "[5,7]" MODEL.TYPE "MLP4"
#python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['linear_784_50', 'relu', 'linear_50_25', 'relu', 'linear_25_25', 'relu', 'linear_25_10']" DATASETS.CLASS_IDS "[8,9]" MODEL.TYPE "MLP4"
#python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['linear_784_50', 'relu', 'linear_50_25', 'relu', 'linear_25_25', 'relu', 'linear_25_10']" DATASETS.CLASS_IDS "[2,4]" MODEL.TYPE "MLP4"
#python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['linear_784_50', 'relu', 'linear_50_25', 'relu', 'linear_25_25', 'relu', 'linear_25_10']" DATASETS.CLASS_IDS "[4,6]" MODEL.TYPE "MLP4"

# Use same class to train a model with 2 conv layers
python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[0,1]" MODEL.TYPE "CNN2"
python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[1,4]" MODEL.TYPE "CNN2"
python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[2,7]" MODEL.TYPE "CNN2"
python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[4,9]" MODEL.TYPE "CNN2"
python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[5,7]" MODEL.TYPE "CNN2"
python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[8,9]" MODEL.TYPE "CNN2"
python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[2,4]" MODEL.TYPE "CNN2"
python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[4,6]" MODEL.TYPE "CNN2"

# Use same class to train a model with 3 conv layers, kernel 3
python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[0,1]" MODEL.TYPE "CNN3"
python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[1,4]" MODEL.TYPE "CNN3"
python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[2,7]" MODEL.TYPE "CNN3"
python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[4,9]" MODEL.TYPE "CNN3"
python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[5,7]" MODEL.TYPE "CNN3"
python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[8,9]" MODEL.TYPE "CNN3"
python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[2,4]" MODEL.TYPE "CNN3"
python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[4,6]" MODEL.TYPE "CNN3"

# Use same class to train a model with 4 conv layers, kernel 3
python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[0,1]" MODEL.TYPE "CNN4"
python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[1,4]" MODEL.TYPE "CNN4"
python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[2,7]" MODEL.TYPE "CNN4"
python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[4,9]" MODEL.TYPE "CNN4"
python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[5,7]" MODEL.TYPE "CNN4"
python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[8,9]" MODEL.TYPE "CNN4"
python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[2,4]" MODEL.TYPE "CNN4"
python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[4,6]" MODEL.TYPE "CNN4"

