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
import time

import warnings
warnings.filterwarnings('ignore')


def get_storage_usage(path):
    list1 = []
    fileList = os.listdir(path)
    for filename in fileList:
        pathTmp = os.path.join(path,filename)  
        if os.path.isdir(pathTmp):   
            get_storage_usage(pathTmp)
        elif os.path.isfile(pathTmp):  
            filesize = os.path.getsize(pathTmp)  
            list1.append(filesize) 
    usage_gb = sum(list1)/1024/1024/1024
    return usage_gb



def check_next_seed(path='results/3_layer_mnist/mnist'):
    max_seed = 0
    for f in os.listdir(path):
        seed = int(f.split('_')[-1].split('.')[0])
        if seed > max_seed:
            max_seed = seed
    return max_seed + 1



def main():

    args = parse_args()
    print(args, end="\n\n")

    output_dir = cfg.SAVE_DIR
    if output_dir:
        mkdir(output_dir)

    

    setproctitle.setproctitle(cfg.NAME)

    mkdir(cfg.TEMP_DIR)

    
    # seed = cfg.SEED
    # if seed == -1:
    #     seed = random.randint(0, 100000)
    if len(os.listdir(cfg.OUTPUT_DIR)) == 0:
        seed = int(time.time()) + random.randint(0, 1000000) #0
    else:
        seed = check_next_seed(path=cfg.OUTPUT_DIR)
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
        filename='_{val_acc:.3f}',
        save_top_k=1,
        monitor='val_acc',
        mode='max',
    )

    callbacks = [checkpoint_callback]

    # create a trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=cfg.SOLVER.GPUS,
        max_epochs=cfg.SOLVER.EPOCHS,
        max_steps=-1,
        log_every_n_steps=50,
        # accumulate_grad_batches=1,
        sync_batchnorm=True,
        strategy="ddp", # ddp_find_unused_parameters_true
        # plugins=DDPPlugin(find_unused_parameters=True),
        num_nodes=1,
        logger=logger,
        # callbacks=callbacks,
        check_val_every_n_epoch=10,
        # val_check_interval=500,
        precision=32,
        # detect_anomaly=True,
    )

    # train the model
    trainer.fit(learner)
    
    print("Training Over")

    pdata = []
    tmp_path = cfg.TEMP_DIR
    for file in glob.glob(os.path.join(tmp_path, "p_data_*.pt")):
        buffers = torch.load(file)
        #for buffer in buffers: # for each saved checkpoint
        #    param = []
        #    for key in buffer.keys():
                # if key in train_layer:
        #        param.append(buffer[key].data) # .reshape(-1) # appends single layer parameters flattened to param list
        #    param = torch.cat(param, 0) # concatenates all layer parameters into single row tensor
        #    pdata.append(param) # append full model flattened to pdata
        pdata.append(buffers)
    #batch = torch.stack(pdata) # [num_models, num_params]
    # mean = torch.mean(batch, dim=0)
    # std = torch.std(batch, dim=0)

    # check the memory of p_data
    # useage_gb = get_storage_usage(tmp_path)
    # print(f"path {tmp_path} storage usage: {useage_gb:.2f} GB")

    state_dic = {
        'pdata': pdata, #batch.cpu().detach(),
        # 'mean': mean.cpu(),
        # 'std': std.cpu(),
        # 'model': torch.load(os.path.join(tmp_path, "whole_model.pth")),
        # 'train_layer': train_layer,
        # 'performance': save_model_accs,
        # 'cfg': config_to_dict(self.cfg)
    }
    
    # Load accuarcy from performance.txt
    acc = []
    with open(os.path.join(tmp_path, "performance.txt"), 'r') as f:
        for line in f:
            acc.append(float(line.strip()))
    perf = acc[-1]
    # Delete the file performance.txt
    os.remove(os.path.join(tmp_path, "performance.txt"))


    # torch.save(state_dic, os.path.join(final_path, "data.pt"))
    classes = str(cfg.DATASETS.CLASS_IDS).replace(" ", "")
    arch = str(cfg.MODEL.TYPE).replace(" ", "")
    saving_name = arch+"_MNIST_"+classes+"_"+str(perf)+'_'+str(seed)+".pt"
    # Check if the file already exists
    if os.path.exists(os.path.join(cfg.OUTPUT_DIR, saving_name)):
        print(f"File {saving_name} already exists")
        saving_name = arch+"_MNIST_"+classes+"_"+str(perf)+'_'+str(seed)+"_copy.pt"

    torch.save(state_dic, os.path.join(cfg.OUTPUT_DIR, saving_name))
    # json_state = {
    #     'cfg': config_to_dict(self.cfg),
    #     'performance': save_model_accs

    # }
    # json.dump(json_state, open(os.path.join(final_path, "config.json"), 'w'))

    # copy the code file(the file) in state_save_dir
    # shutil.copy(os.path.abspath(__file__), os.path.join(final_path, os.path.basename(__file__)))

    # delete the tmp_path
    try:
        shutil.rmtree(tmp_path)
        print("Saving process over")
    except:
        pass


if __name__ == '__main__':
    main()





