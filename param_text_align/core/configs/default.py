from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.LAYER_LIST = ["linear_3072_100", "relu", "linear_100_10"]
# _C.MODEL.NAME = "deeplabv3plus_resnet101"
# _C.MODEL.NUM_CLASSES = 19
# _C.MODEL.WEIGHTS = ""
# _C.MODEL.FREEZE_BN = True
# _C.MODEL.HYPER = True
# _C.MODEL.CURVATURE = 1.0
# _C.MODEL.REDUCED_CHANNELS = 64
# _C.MODEL.HFR = True

_C.WANDB = CN()
_C.WANDB.ENABLE = False
_C.WANDB.GROUP = "deeplabv2_r101_pretrain"
_C.WANDB.PROJECT = "active_domain_adapt"
_C.WANDB.ENTITY = "pinlab-sapienza"

_C.INPUT = CN()
_C.INPUT.SIZE = (10, 785)
_C.INPUT.IN_CHANNELS = 1
_C.INPUT.OUT_CHANNELS = 1
_C.INPUT.TIME_DIM = 256

_C.DATASETS = CN()
_C.DATASETS.TRAIN = "checkpoints/mnist_mlp_1layer"
_C.DATASETS.TEST = ""

_C.SOLVER = CN()
_C.SOLVER.GPUS = [0, 1, 2, 3]
_C.SOLVER.NUM_WORKERS = 8
_C.SOLVER.BASE_LR = 1e-3
_C.SOLVER.BATCH_SIZE = 1

# optimizer and learning rate
_C.SOLVER.LR_METHOD = "poly"
_C.SOLVER.LR_POWER = 0.5
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WARMUP_ITERS = 600
_C.SOLVER.BATCH_SIZE_VAL = 1


_C.DIFFUSION = CN()
# active strategy
_C.DIFFUSION.NOISE_STEP=1000
_C.DIFFUSION.BETA_START = 1e-4
_C.DIFFUSION.BETA_END = 0.02

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 1

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.NAME = "debug"
_C.OUTPUT_DIR = "results/debug"
_C.resume = ""
_C.SEED = -1
_C.DEBUG = 0