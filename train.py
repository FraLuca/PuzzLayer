import os
import random
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
# from pytorch_lightning.plugins import DDPPlugin

import setproctitle
from core.learner import Learner
from core.utils.misc import mkdir, parse_args
from core.configs import cfg

import glob
import torch
import shutil

import warnings
warnings.filterwarnings('ignore')
# os.environ["NCCL_P2P_DISABLE"] = "1"

class PeriodicCheckpoint(ModelCheckpoint):
    def __init__(self, dirpath: str, every: int):
        super().__init__()
        self.dirpath = dirpath
        self.every = every

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs
    ):
        if (pl_module.current_epoch + 1) % self.every == 0:
            assert self.dirpath is not None
            self.filename = f"model_{pl_module.current_epoch}.ckpt"
            filepath = self.dirpath + "/" + self.filename
            trainer.save_checkpoint(filepath)

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

    # create a logger
    logger = None
    if cfg.WANDB.ENABLE:
        logger = WandbLogger(
            project=cfg.WANDB.PROJECT,
            name=cfg.NAME,
            entity=cfg.WANDB.ENTITY,
            group=cfg.WANDB.GROUP,
            config=cfg,
            save_dir=".",
        )

    # create a checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.SAVE_DIR,
        filename='_{val_recall_t2i@1:.3f}',
        save_top_k=1,
        monitor='val_recall_t2i@1',
        mode='max',
    )
    periodic_checkpoint = PeriodicCheckpoint(cfg.SAVE_DIR, cfg.SAVE_CHECK_EVERY)

    callbacks = [checkpoint_callback]

    # create a trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=cfg.SOLVER.GPUS,
        max_epochs=cfg.SOLVER.EPOCHS,
        max_steps=-1,
        log_every_n_steps=50,
        accumulate_grad_batches=4,
        sync_batchnorm=True,
        strategy="ddp_find_unused_parameters_true", # ddp_find_unused_parameters_true
        # plugins=DDPPlugin(find_unused_parameters=True),
        num_nodes=1,
        logger=logger,
        callbacks=callbacks,
        check_val_every_n_epoch=1,
        # val_check_interval=500,
        precision=16,
        # detect_anomaly=True,
    )

    # train the model
    trainer.fit(learner)
    
    print("Training Over")



if __name__ == '__main__':
    main()

