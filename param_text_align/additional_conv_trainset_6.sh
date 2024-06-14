for i in `seq 0 300`; do
{
    python train.py -cfg configs/configs_for_generations/train_6.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[1,5]" MODEL.TYPE "CNN4"
    python train.py -cfg configs/configs_for_generations/train_6.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[1,6]" MODEL.TYPE "CNN4"
    python train.py -cfg configs/configs_for_generations/train_6.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[1,7]" MODEL.TYPE "CNN4"
    python train.py -cfg configs/configs_for_generations/train_6.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[1,8]" MODEL.TYPE "CNN4"
    python train.py -cfg configs/configs_for_generations/train_6.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[1,9]" MODEL.TYPE "CNN4"
    python train.py -cfg configs/configs_for_generations/train_6.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[2,4]" MODEL.TYPE "CNN4"
    python train.py -cfg configs/configs_for_generations/train_6.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[2,5]" MODEL.TYPE "CNN4"
    python train.py -cfg configs/configs_for_generations/train_6.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[2,6]" MODEL.TYPE "CNN4"
    python train.py -cfg configs/configs_for_generations/train_6.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[2,7]" MODEL.TYPE "CNN4"
    python train.py -cfg configs/configs_for_generations/train_6.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[2,8]" MODEL.TYPE "CNN4"
    python train.py -cfg configs/configs_for_generations/train_6.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[2,9]" MODEL.TYPE "CNN4"
    python train.py -cfg configs/configs_for_generations/train_6.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[3,5]" MODEL.TYPE "CNN4"
    python train.py -cfg configs/configs_for_generations/train_6.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[3,6]" MODEL.TYPE "CNN4"
}
done