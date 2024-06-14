import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
# from tqdm.auto import tqdm

from core.configs import cfg
from core.datasets.build import build_dataset, build_ae_dataset, build_ae_dataset_flat

from core.models.build import build_model, MldVae, MldAe, init_model
from core.models.build2 import medium

from core.models.vae_utils.losses.kl import KLLoss

import os

def filter_classes(dataset, classes):
    indices = [i for i, (_, label) in enumerate(dataset) if label in classes]
    return Subset(dataset, indices)



class Learner(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = build_model(cfg)
        self.model = init_model(cfg, self.model)
        print(self.model)
        self.criterion = nn.CrossEntropyLoss()

        self.save_hyperparameters(cfg)
        self.parameters_list = []

    def forward(self, x):
        return self.model(x)

    def state_part(self, net):
        part_param = {}
        for name, param in net.named_parameters():
            part_param[name] = param.detach().cpu()
        return part_param

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        accuracy = (y_hat.argmax(1) == y).float().mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        accuracy = (y_hat.argmax(1) == y).float().mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # Save accuuracy into a file
        with open(os.path.join(cfg.TEMP_DIR, "performance.txt"), "a") as f:
            acc_rounded = round(accuracy.item(), 2)
            f.write(f"{acc_rounded}\n")
        return loss
    
    def train_dataloader(self):
        train_set = build_dataset(self.cfg, train=True)
        if self.cfg.DATASETS.CLASS_IDS != "all":
            train_set = filter_classes(train_set, self.cfg.DATASETS.CLASS_IDS)
        
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=self.cfg.SOLVER.BATCH_SIZE,
            shuffle=True,
            num_workers=self.cfg.SOLVER.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,)
        return train_loader

    def val_dataloader(self):
        val_set = build_dataset(self.cfg, train=False)
        if self.cfg.DATASETS.CLASS_IDS != "all":
            val_set = filter_classes(val_set, self.cfg.DATASETS.CLASS_IDS)
        
        val_loader = DataLoader(
            dataset=val_set,
            batch_size=self.cfg.SOLVER.BATCH_SIZE_VAL,
            shuffle=False,
            num_workers=self.cfg.SOLVER.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,)
        return val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY, momentum=cfg.SOLVER.MOMENTUM)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.SOLVER.EPOCHS, eta_min=cfg.SOLVER.ETA_MIN)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.SOLVER.MILESTONE, gamma=0.2)

        return [optimizer], [scheduler] 
    
    def on_train_epoch_end(self):
        if self.current_epoch >= self.cfg.MODEL.SAVE_START and self.current_epoch % 10 == 0:
            # print(self.current_epoch, len(self.parameters_list))
            torch.save(self.model, os.path.join(cfg.TEMP_DIR, "p_data_{}.pt".format(self.current_epoch)))
            #self.parameters_list.append(self.state_part(self.model))
            #if len(self.parameters_list) == 10 or (self.current_epoch == self.cfg.SOLVER.EPOCHS-1):
            #    torch.save(self.parameters_list, os.path.join(cfg.TEMP_DIR, "p_data_{}.pt".format(self.current_epoch)))
            #    self.parameters_list = []
        else:
            pass







class VAELearner(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # self.vae = MldVae(nfeats=1024, latent_dim=[1, 32], ff_size=64, num_layers=3, position_embedding='learned') # nfeats=D
        # self.vae = MldAe(nfeats=1, latent_dim=[1, 32], ff_size=64, num_layers=3, position_embedding='learned') # nfeats=D
        self.vae = medium(65536, 0.1, 0.1)
        # self.encoder, self.decoder = build_vae_model(cfg)
        # self.criterion = nn.MSELoss()

        # self.save_hyperparameters(cfg)

        self.l1_loss = nn.SmoothL1Loss(reduction="mean")
        # self.kl_loss = KLLoss()
        self.lambda_kl = 1.e-4

    def forward(self, x, lengths):
        # latent, dist = self.vae.encode(x, lengths)
        dist = None
        # latent = self.vae.encode(x, lengths)
        # reconstructed =  self.vae.decode(latent, lengths)

        reconstructed = self.vae(x).squeeze(1)

        if dist is not None:
            # Create a centred normal distribution to compare with
            mu_ref = torch.zeros_like(dist.loc)
            scale_ref = torch.ones_like(dist.scale)
            dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
        else:
            dist_ref = dist

        return reconstructed, dist, dist_ref

    def training_step(self, batch, batch_idx):
        x, N, y, arch = batch
        # x = x.view(x.size(0), -1, 1)
        # lengths = [x.size(1)]
        reconstructed, dist, dist_ref = self(x, N)
        # reconstructed = reconstructed[:, :N]

        recon_loss = self.l1_loss(reconstructed[:, :N], x.squeeze(1)[:, :N])
        # kl_div = self.kl_loss(dist, dist_ref)

        loss = recon_loss #+ self.lambda_kl * kl_div

        self.log('train_recon_loss', recon_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        # self.log('train_kl_loss', kl_div, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, N, y, arch = batch
        # lengths = [x.size(1)]
        reconstructed, dist, dist_ref = self(x, N)

        recon_loss = self.l1_loss(reconstructed[:, :N], x.squeeze(1)[:, :N])
        # kl_div = self.kl_loss(dist, dist_ref)

        loss = recon_loss #+ self.lambda_kl * kl_div

        self.log('val_recon_loss', recon_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # self.log('val_kl_loss', kl_div, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss
    
    def train_dataloader(self):
        train_set = build_ae_dataset_flat(self.cfg, train=True)
        
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=self.cfg.SOLVER.BATCH_SIZE,
            shuffle=True,
            num_workers=self.cfg.SOLVER.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,)
        return train_loader

    def val_dataloader(self):
        val_set = build_ae_dataset_flat(self.cfg, train=False)
        
        val_loader = DataLoader(
            dataset=val_set,
            batch_size=self.cfg.SOLVER.BATCH_SIZE_VAL,
            shuffle=False,
            num_workers=self.cfg.SOLVER.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,)
        return val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        schduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.SOLVER.EPOCHS, eta_min=cfg.SOLVER.ETA_MIN)

        return [optimizer], [schduler] 







from core.datasets.build import list_of_parms
from core.models.build import extract_layers_from_path

class TestVAELearner(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.l1_loss = nn.SmoothL1Loss(reduction="mean")
        self.kl_loss = KLLoss()
        self.criterion = nn.CrossEntropyLoss()

        self.vae = MldAe(nfeats=1024, latent_dim=[1, 32], ff_size=64, num_layers=3, position_embedding='learned') # nfeats=D
        # load pretrained VAE from cfg.resume_vae
        vae_state_dict = torch.load(cfg.resume_vae)['state_dict']
        vae_state_dict = {k.replace('vae.', ''): v for k, v in vae_state_dict.items()}
        # remove keys that starts with model.
        vae_state_dict = {k: v for k, v in vae_state_dict.items() if not k.startswith('model.')}
        self.vae.load_state_dict(vae_state_dict)

        # build instances of original version and reconstructed version of the model
        layer_list = extract_layers_from_path(path=cfg.resume_model)
        self.model_orig = build_model(layer_list=layer_list)
        # load the model from the checkpoint at cfg.resume_model
        pretrained_state_dict = torch.load(cfg.resume_model)['state_dict']
        # remove 'model.' from the keys of the state_dict
        pretrained_state_dict = {k.replace('model.', ''): v for k, v in pretrained_state_dict.items()}
        self.model_orig.load_state_dict(pretrained_state_dict)

        # instanciate the reconstructed model with the same architecture of the original model
        self.model_recon = build_model(layer_list=layer_list)
        print(self.model_orig)

        params_list = list_of_parms(path=cfg.resume_model)
        recon_loss = 0
        kl_div = 0

        recon_state_dict = {}

        for param in params_list:
            x, N, y, arch = param # forse bisogna fare unsqueeze per la batch dimension
            reconstructed, dist, dist_ref = self.vae_inference(x.unsqueeze(0), [N])

            recon_loss += self.l1_loss(reconstructed, x.unsqueeze(0))
            # kl_div += self.kl_loss(dist, dist_ref)

            reconstructed = reconstructed.squeeze(0)

            orig_param = self.model_orig.state_dict()[y]

            if 'bias' in y or 'batchnorm' in y:
                orig_param_shape = orig_param.reshape(1, -1).shape
            elif 'weight' in y:
                orig_param_shape = orig_param.reshape(orig_param.shape[0], -1).shape

            # remove pad from reconstruted and then reshape to the original shape
            reconstructed = reconstructed[:, :orig_param_shape[1]]
            reconstructed = reconstructed.reshape(orig_param.shape)

            recon_state_dict[y] = reconstructed

        # add not learned parameters from the original model
        for k, v in self.model_orig.state_dict().items():
            if k not in recon_state_dict:
                recon_state_dict[k] = v

        # load the reconstructed parameter in the layer y of the model self.model_recon
        self.model_recon.load_state_dict(recon_state_dict) # TODO


        # Metrics
        self.recon_loss = recon_loss / len(params_list)
        # self.kl_div = kl_div / len(params_list)

        self.loss_orig = 0
        self.loss_recon = 0
        self.acc_orig = 0
        self.acc_recon = 0


    def vae_inference(self, x, lengths):
        # latent, dist = self.vae.encode(x, lengths)
        latent = self.vae.encode(x, lengths)
        dist = None
        reconstructed =  self.vae.decode(latent, lengths)

        if dist is not None:
            # Create a centred normal distribution to compare with
            mu_ref = torch.zeros_like(dist.loc)
            scale_ref = torch.ones_like(dist.scale)
            dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
        else:
            dist_ref = dist

        return reconstructed, dist, dist_ref


    def test_step(self, batch, batch_idx):

        x, y = batch

        y_hat_orig = self.model_orig(x)
        loss_orig = self.criterion(y_hat_orig, y)
        accuracy_orig = (y_hat_orig.argmax(1).reshape(1) == y).float().mean()

        y_hat_recon = self.model_recon(x)
        loss_recon = self.criterion(y_hat_recon, y)
        accuracy_recon = (y_hat_recon.argmax(1) == y).float().mean()

        self.loss_orig += loss_orig
        self.loss_recon += loss_recon
        self.acc_orig += accuracy_orig
        self.acc_recon += accuracy_recon

        # print(f"Orig loss: {loss_orig/10000}, Recon loss: {loss_recon/10000}")
        # print(f"Orig acc: {accuracy_orig/10000}, Recon acc: {accuracy_recon/10000}")



    def on_test_epoch_end(self):

        self.log('loss_orig', self.loss_orig/10000, on_step=False, on_epoch=True, prog_bar=True)
        self.log('loss_recon', self.loss_recon/10000, on_step=False, on_epoch=True, prog_bar=True)
        self.log('acc_orig', 100*self.acc_orig/10000, on_step=False, on_epoch=True, prog_bar=True)
        self.log('acc_recon', 100*self.acc_recon/10000, on_step=False, on_epoch=True, prog_bar=True)
        self.log('VAE_recon_loss', self.recon_loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log('VAE_kl_div', self.kl_div, on_step=False, on_epoch=True, prog_bar=True)



    # def training_step(self, batch, batch_idx):
    #     x, N, y, arch = batch
    #     # x = x.view(x.size(0), -1, 1)
    #     # lengths = [x.size(1)]
    #     reconstructed, dist, dist_ref = self(x, N)

    #     recon_loss = self.l1_loss(reconstructed, x)
    #     kl_div = self.kl_loss(dist, dist_ref)

    #     loss = recon_loss + self.lambda_kl * kl_div

    #     self.log('train_recon_loss', recon_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
    #     self.log('train_kl_loss', kl_div, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
    #     self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
    #     return loss
    
    # def validation_step(self, batch, batch_idx):
    #     x, N, y, arch = batch
    #     # lengths = [x.size(1)]
    #     reconstructed, dist, dist_ref = self(x, N)

    #     recon_loss = self.l1_loss(reconstructed, x)
    #     kl_div = self.kl_loss(dist, dist_ref)

    #     loss = recon_loss + self.lambda_kl * kl_div

    #     self.log('val_recon_loss', recon_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
    #     self.log('val_kl_loss', kl_div, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
    #     self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
    #     return loss

    def test_dataloader(self):
        test_set = build_dataset(self.cfg, train=False)
        
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=self.cfg.SOLVER.BATCH_SIZE_VAL,
            shuffle=False,
            num_workers=self.cfg.SOLVER.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,)
        return test_loader

    
    # def train_dataloader(self):
    #     train_set = build_ae_dataset(self.cfg, train=True)
        
    #     train_loader = DataLoader(
    #         dataset=train_set,
    #         batch_size=self.cfg.SOLVER.BATCH_SIZE,
    #         shuffle=True,
    #         num_workers=self.cfg.SOLVER.NUM_WORKERS,
    #         pin_memory=True,
    #         drop_last=True,
    #         persistent_workers=True,)
    #     return train_loader

    # def val_dataloader(self):
    #     val_set = build_ae_dataset(self.cfg, train=False)
        
    #     val_loader = DataLoader(
    #         dataset=val_set,
    #         batch_size=self.cfg.SOLVER.BATCH_SIZE_VAL,
    #         shuffle=False,
    #         num_workers=self.cfg.SOLVER.NUM_WORKERS,
    #         pin_memory=True,
    #         persistent_workers=True,)
    #     return val_loader


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        schduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.SOLVER.EPOCHS, eta_min=cfg.SOLVER.ETA_MIN)

        return [optimizer], [schduler] 