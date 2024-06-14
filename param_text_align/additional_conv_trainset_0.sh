for i in `seq 0 300`; do
{
    python train.py -cfg configs/configs_for_generations/train_0.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[0,2]" MODEL.TYPE "CNN2"
    python train.py -cfg configs/configs_for_generations/train_0.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[0,3]" MODEL.TYPE "CNN2"
    python train.py -cfg configs/configs_for_generations/train_0.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[0,4]" MODEL.TYPE "CNN2"
    python train.py -cfg configs/configs_for_generations/train_0.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[0,5]" MODEL.TYPE "CNN2"
    python train.py -cfg configs/configs_for_generations/train_0.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[0,6]" MODEL.TYPE "CNN2"
    python train.py -cfg configs/configs_for_generations/train_0.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[0,7]" MODEL.TYPE "CNN2"
    python train.py -cfg configs/configs_for_generations/train_0.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[0,8]" MODEL.TYPE "CNN2"
    python train.py -cfg configs/configs_for_generations/train_0.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[1,3]" MODEL.TYPE "CNN2"
    python train.py -cfg configs/configs_for_generations/train_0.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[1,4]" MODEL.TYPE "CNN2"
    python train.py -cfg configs/configs_for_generations/train_0.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[1,5]" MODEL.TYPE "CNN2"
    python train.py -cfg configs/configs_for_generations/train_0.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[1,6]" MODEL.TYPE "CNN2"
    python train.py -cfg configs/configs_for_generations/train_0.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[1,7]" MODEL.TYPE "CNN2"
    python train.py -cfg configs/configs_for_generations/train_0.yaml MODEL.LAYER_LIST "['conv_1_32_3', 'relu', 'conv_32_32_3', 'relu', 'adapool_1', 'linear_32_10']" DATASETS.CLASS_IDS "[1,8]" MODEL.TYPE "CNN2"
}
done