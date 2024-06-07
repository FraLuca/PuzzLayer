import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from core.model.build import build_model, init_model
from core.model.model_encoder import ModelEncoder
from core.model.text_encoder import TextEncoder
from core.dataset.build import build_dataset, custom_collate_fn
from core.model.utils.loss import CLIPLoss
from core.configs import cfg

from torch_geometric.data import Batch
from transformers import BertTokenizer, get_linear_schedule_with_warmup


class Learner(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_encoder = ModelEncoder()

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        vocab_size = self.tokenizer.vocab_size
        self.text_encoder = TextEncoder(vocab_size)

        self.criterion = CLIPLoss()

        self.save_hyperparameters(cfg)



    def forward(self, model_batch, text_batch):
        model_embed = self.model_encoder(model_batch)
        text_embed = self.text_encoder(text_batch)
        return model_embed, text_embed

    def training_step(self, batch, batch_idx):
        model_batch, text_batch = batch

        # model_batch = Batch.from_data_list(model_batch)
        text_batch = torch.tensor([self.tokenizer.encode(t) for t in text_batch]).to(model_batch.x.device)

        model_embed, text_embed = self(model_batch, text_batch)
        
        loss = self.criterion(model_embed, text_embed)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
                
        return loss

    def validation_step(self, batch, batch_idx):
        model_batch, text_batch = batch

        # model_batch = Batch.from_data_list(model_batch)
        text_batch = torch.tensor([self.tokenizer.encode(t) for t in text_batch]).to(model_batch.x.device)

        model_embed, text_embed = self(model_batch, text_batch)
        
        loss = self.criterion(model_embed, text_embed)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss


    def train_dataloader(self):
        train_set = build_dataset(self.cfg, train=True)

        train_loader = DataLoader(
            dataset=train_set,
            batch_size=self.cfg.SOLVER.BATCH_SIZE,
            shuffle=True,
            num_workers=self.cfg.SOLVER.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            collate_fn=custom_collate_fn,
            )
        
        return train_loader

    def val_dataloader(self):
        val_set = build_dataset(self.cfg, train=False)
        
        val_loader = DataLoader(
            dataset=val_set,
            batch_size=self.cfg.SOLVER.BATCH_SIZE_VAL,
            shuffle=False,
            num_workers=self.cfg.SOLVER.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=custom_collate_fn,
            )
        return val_loader


    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY, momentum=cfg.SOLVER.MOMENTUM)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.SOLVER.MILESTONE, gamma=0.2)

        optimizer = torch.optim.AdamW( list(self.model_encoder.parameters())+list(self.text_encoder.parameters()), lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=cfg.SOLVER.WARMUP_ITERS, num_training_steps=-1)

        return [optimizer], [scheduler]