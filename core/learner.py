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
from torch.optim.lr_scheduler import LinearLR
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from core.model.utils.metrics import *
from core.model.head import ProjectionHead


class Learner(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_encoder = ModelEncoder(input_dim=cfg.MODEL.MODEL_INPUT_DIM,
                                          output_dim=cfg.MODEL.MODEL_OUTPUT_DIM,
                                          dropout=0.2)

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        vocab_size = self.tokenizer.vocab_size
        self.text_encoder = TextEncoder(vocab_size,
                                        input_dim=cfg.MODEL.TEXT_INPUT_DIM,
                                        num_heads=2,
                                        num_layers=2,
                                        dropout=cfg.MODEL.DROPOUT)

        self.text_projection = ProjectionHead(embedding_dim=cfg.MODEL.TEXT_INPUT_DIM,
                                              projection_dim=cfg.MODEL.PROJ_OUTPUT_DIM,
                                              dropout=cfg.MODEL.DROPOUT)
        
        self.model_projection = ProjectionHead(embedding_dim=cfg.MODEL.MODEL_OUTPUT_DIM,
                                                projection_dim=cfg.MODEL.PROJ_OUTPUT_DIM,
                                                dropout=cfg.MODEL.DROPOUT)

        self.criterion = CLIPLoss()

        if cfg.PRETRAINED_MODEL_ENCODER:
            print(f"Loading pretrained model encoder from {cfg.PRETRAINED_MODEL_ENCODER}")
            self.load_checkpoint(cfg.PRETRAINED_MODEL_ENCODER)

        self.save_hyperparameters(cfg)


    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        pre_weights = {k: v for k, v in checkpoint["state_dict"].items() if "model_encoder" in k}
        self.load_state_dict(pre_weights, strict=False)


    def forward(self, model_batch, text_batch, f=None):
        model_embed = self.model_projection(self.model_encoder(model_batch, f))
        text_embed = self.text_projection(self.text_encoder(text_batch))
        return model_embed, text_embed

    def training_step(self, batch, batch_idx):
        model_batch, text_batch, f = batch

        text_batch = torch.tensor([self.tokenizer.encode(t) for t in text_batch]).to(model_batch.x.device)[:, 1:-1]

        model_embed, text_embed = self(model_batch, text_batch, f)
        
        loss = self.criterion(model_embed, text_embed)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        sim = self.compute_sim_matrix(model_embed, text_embed)
        # acc = self.compute_accuracy_alignment(model_embed, text_embed)
        # self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        for k in [1, 3]:
            recall_i2t, recall_t2i = recall_at_k(sim, k)
            self.log(f"train_recall_i2t@{k}", recall_i2t, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"train_recall_t2i@{k}", recall_t2i, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        model_batch, text_batch, f = batch

        text_batch = torch.tensor([self.tokenizer.encode(t) for t in text_batch]).to(model_batch.x.device)[:, 1:-1]

        model_embed, text_embed = self(model_batch, text_batch, f)
        
        loss = self.criterion(model_embed, text_embed)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        sim = self.compute_sim_matrix(model_embed, text_embed, f)
        # acc = self.compute_accuracy_alignment(model_embed, text_embed, f)
        # self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        for k in [1, 3]:
            recall_i2t, recall_t2i = recall_at_k(sim, k)
            self.log(f"val_recall_i2t@{k}", recall_i2t, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"val_recall_t2i@{k}", recall_t2i, on_step=False, on_epoch=True, sync_dist=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        model_batch, text_batch, f = batch

        text_batch = torch.tensor([self.tokenizer.encode(t) for t in text_batch]).to(model_batch.x.device)[:, 1:-1]

        model_embed, text_embed = self(model_batch, text_batch, f)
        
        loss = self.criterion(model_embed, text_embed)

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        sim = self.compute_sim_matrix(model_embed, text_embed, f)
        # acc = self.compute_accuracy_alignment(model_embed, text_embed, f)
        # self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        for k in [1, 3]:
            recall_i2t, recall_t2i = recall_at_k(sim, k)
            self.log(f"test_recall_i2t@{k}", recall_i2t, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"test_recall_t2i@{k}", recall_t2i, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def compute_sim_matrix(self, model_features, text_features, f=None):
        model_features = F.normalize(model_features, dim=1)
        text_features = F.normalize(text_features, dim=1)

        # compute cosine similarity
        sim = model_features @ text_features.t()

        return sim


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
    
    def test_dataloader(self):
        test_set = build_dataset(self.cfg, train=False)
        
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=self.cfg.SOLVER.BATCH_SIZE_VAL,
            shuffle=False,
            num_workers=self.cfg.SOLVER.NUM_WORKERS,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=custom_collate_fn,
            )
        return test_loader


    def configure_optimizers(self):
        
        parameters = [
            {"params": self.model_encoder.parameters(), "lr": self.cfg.SOLVER.MODEL_ENCODER_LR},
            {"params": self.text_encoder.parameters(), "lr": self.cfg.SOLVER.TEXT_ENCODER_LR},
            {
                "params": list(self.model_projection.parameters()) + list(self.text_projection.parameters()),
                "lr": self.cfg.SOLVER.PROJ_LR,
                "weight_decay": self.cfg.SOLVER.WEIGHT_DECAY,
            },
        ]
        optimizer = torch.optim.AdamW(parameters, weight_decay=self.cfg.SOLVER.WEIGHT_DECAY)

        return {
            "optimizer": optimizer,
        }