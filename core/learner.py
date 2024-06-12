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

        # self.criterion = CLIPLoss()
        self.criterion = nn.CrossEntropyLoss()

        self.classifier = nn.Linear(cfg.MODEL.OUTPUT_DIM, 9)

        self.save_hyperparameters(cfg)
        # self.automatic_optimization = False



    def forward(self, model_batch, text_batch, f=None):
        model_embed = self.model_encoder(model_batch, f)
        # text_embed = self.text_encoder(text_batch)
        return model_embed #, text_embed

    def training_step(self, batch, batch_idx):
        # opt1 = self.optimizers()
        # opt1.zero_grad()
        # scheduler1 = self.lr_schedulers()

        model_batch, text_batch, f = batch

        ## model_batch = Batch.from_data_list(model_batch)
        # text_batch = torch.tensor([self.tokenizer.encode(t) for t in text_batch]).to(model_batch.x.device)

        # model_embed, text_embed = self(model_batch, text_batch, f)
        model_embed = self(model_batch, text_batch, f)
        class_logits = self.classifier(model_embed)
        
        # loss = self.criterion(model_embed, text_embed)
        loss = self.criterion(class_logits, text_batch)
        acc = (class_logits.argmax(dim=1) == text_batch).float().mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # self.manual_backward(loss)
        # opt1.step()
        # scheduler1.step()
        
        return loss

    def validation_step(self, batch, batch_idx):
        model_batch, text_batch, f = batch

        ## model_batch = Batch.from_data_list(model_batch)
        # text_batch = torch.tensor([self.tokenizer.encode(t) for t in text_batch]).to(model_batch.x.device)

        # model_embed, text_embed = self(model_batch, text_batch, f)
        model_embed = self(model_batch, text_batch, f)
        class_logits = self.classifier(model_embed)
        
        # loss = self.criterion(model_embed, text_embed)
        loss = self.criterion(class_logits, text_batch)
        acc = (class_logits.argmax(dim=1) == text_batch).float().mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

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

        optimizer1 = torch.optim.AdamW(self.model_encoder.parameters(), lr=cfg.SOLVER.BASE_LR1, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        # optimizer2 = torch.optim.AdamW(self.classifier.parameters(), lr=cfg.SOLVER.BASE_LR2, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        scheduler1 = get_linear_schedule_with_warmup(optimizer1, num_warmup_steps=cfg.SOLVER.WARMUP_ITERS, num_training_steps=-1)
        # scheduler2 = get_linear_schedule_with_warmup(optimizer2, num_warmup_steps=cfg.SOLVER.WARMUP_ITERS, num_training_steps=-1)

        return [optimizer1], [scheduler1]