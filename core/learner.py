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


class Learner(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_encoder = ModelEncoder()

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        vocab_size = self.tokenizer.vocab_size
        self.text_encoder = TextEncoder(vocab_size, embed_dim=64, num_heads=1, num_layers=2, dropout=0.2)

        self.criterion = CLIPLoss()
        # self.criterion = nn.CrossEntropyLoss()

        # self.classifier = nn.Sequential(nn.ReLU(), nn.Linear(cfg.MODEL.OUTPUT_DIM, cfg.MODEL.OUTPUT_DIM//2),
        #                                 nn.ReLU(), nn.Linear(cfg.MODEL.OUTPUT_DIM//2, cfg.MODEL.NUM_CLASSES))

        if cfg.PRETRAINED_MODEL_ENCODER:
            self.load_checkpoint(cfg.PRETRAINED_MODEL_ENCODER)

        self.save_hyperparameters(cfg)
        # self.automatic_optimization = False

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_encoder_weights = {k: v for k, v in checkpoint["state_dict"].items() if "model_encoder" in k}
        self.load_state_dict(model_encoder_weights, strict=False)


    def forward(self, model_batch, text_batch, f=None):
        model_embed = self.model_encoder(model_batch, f)
        text_embed = self.text_encoder(text_batch)
        return model_embed, text_embed

    def training_step(self, batch, batch_idx):
        # opt1 = self.optimizers()
        # opt1.zero_grad()
        # scheduler1 = self.lr_schedulers()

        model_batch, text_batch, f = batch

        # model_batch = Batch.from_data_list(model_batch)
        text_batch = torch.tensor([self.tokenizer.encode(t) for t in text_batch]).to(model_batch.x.device)[:, 1:3]

        model_embed, text_embed = self(model_batch, text_batch, f)
        # model_embed = self(model_batch, text_batch, f)
        # class_logits = self.classifier(model_embed)
        
        loss = self.criterion(model_embed, text_embed)
        # loss = self.criterion(class_logits, text_batch.float())

        # compute accuracy for multi label classification (2 classes)
        # acc = torch.topk(class_logits, 2).indices
        # covert acc to one hot
        # acc = torch.zeros_like(class_logits).scatter(1, acc, 1)
        # acc = accuracy_score(text_batch.cpu().numpy(), acc.cpu().numpy())
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # select only indices of the list f that contain "CNN2"
        cnn2_indices = [i for i in range(len(f)) if "CNN2" in f[i]]
        acc = self.compute_accuracy_alignment(model_embed[cnn2_indices], text_embed[cnn2_indices])
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # self.manual_backward(loss)
        # opt1.step()
        # scheduler1.step()
        
        return loss

    def validation_step(self, batch, batch_idx):
        model_batch, text_batch, f = batch

        # model_batch = Batch.from_data_list(model_batch)
        text_batch = torch.tensor([self.tokenizer.encode(t) for t in text_batch]).to(model_batch.x.device)[:, 1:3]

        model_embed, text_embed = self(model_batch, text_batch, f)
        # model_embed = self(model_batch, text_batch, f)
        # class_logits = self.classifier(model_embed)
        
        loss = self.criterion(model_embed, text_embed)
        # loss = self.criterion(class_logits, text_batch.float())

        # acc = torch.topk(class_logits, 2).indices
        # covert acc to one hot
        # acc = torch.zeros_like(class_logits).scatter(1, acc, 1)
        # acc = accuracy_score(text_batch.cpu().numpy(), acc.cpu().numpy())

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # select only indices of the list f that contain "CNN2"
        cnn2_indices = [i for i in range(len(f)) if "CNN2" in f[i]]
        acc = self.compute_accuracy_alignment(model_embed[cnn2_indices], text_embed[cnn2_indices])
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def compute_accuracy_alignment(self, model_features, text_features):
        model_features = F.normalize(model_features, dim=1)
        text_features = F.normalize(text_features, dim=1)

        # compute cosine similarity
        sim = text_features @ model_features.t()
        acc = (torch.argmax(sim, dim=1) == torch.arange(sim.shape[0], device=sim.device)).float().mean()

        return acc

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

        list_parameters = list(self.model_encoder.parameters()) + list(self.text_encoder.parameters()) #+ list(self.classifier.parameters())
        optimizer1 = torch.optim.AdamW(list_parameters, lr=cfg.SOLVER.BASE_LR1, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        # optimizer2 = torch.optim.AdamW(self.classifier.parameters(), lr=cfg.SOLVER.BASE_LR2, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        # scheduler1 = get_linear_schedule_with_warmup(optimizer1, num_warmup_steps=cfg.SOLVER.WARMUP_ITERS, num_training_steps=-1)
        # scheduler2 = get_linear_schedule_with_warmup(optimizer2, num_warmup_steps=cfg.SOLVER.WARMUP_ITERS, num_training_steps=-1)
        # linear_sched = LinearLR(optimizer1, start_factor=0.05, total_iters=9600)

        return [optimizer1], []