import random
import pytorch_lightning as pl

import setproctitle
from core.learner import Learner
from core.utils.misc import mkdir, parse_args
from core.configs import cfg

import glob
import torch
import shutil

import warnings
warnings.filterwarnings('ignore')

def main():

    args = parse_args()
    print(args, end="\n\n")

    output_dir = cfg.SAVE_DIR
    if output_dir:
        mkdir(output_dir)

    
    setproctitle.setproctitle(cfg.NAME)
    
    seed = cfg.SEED
    if seed == -1:
        seed = random.randint(0, 100000)
    pl.seed_everything(seed, workers=True)


    # create a learner that create a model, a dataloader and a loss function
    learner = Learner(cfg)

    # create a trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=cfg.SOLVER.GPU_TEST,
        max_epochs=cfg.SOLVER.EPOCHS,
        max_steps=-1,
        log_every_n_steps=50,
        # accumulate_grad_batches=4,
        sync_batchnorm=True,
        # strategy="ddp_find_unused_parameters_true", # ddp_find_unused_parameters_true
        # plugins=DDPPlugin(find_unused_parameters=True),
        num_nodes=1,
        logger=None,
        check_val_every_n_epoch=1,
        # val_check_interval=500,
        precision=32,
        # detect_anomaly=True,
        # deterministic=True
    )

    # train the model
    trainer.test(learner, ckpt_path=cfg.MODEL_TO_TEST)
    
    print("Testing Over")



if __name__ == '__main__':
    main()