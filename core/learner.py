import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from core.model.build import build_model, init_model
from core.model.model_encoder import ModelEncoder
from core.model.text_encoder import TextEncoder, TextEncoderCustom
from core.model.model_decoder import ModelDecoder
from core.dataset.build import build_dataset, custom_collate_fn
from core.model.utils.loss import CLIPLoss
from core.configs import cfg

from torch_geometric.data import Batch
from transformers import BertTokenizer, get_linear_schedule_with_warmup, BertModel
from torch.optim.lr_scheduler import LinearLR
from sklearn.metrics import accuracy_score
from torch_geometric.data import Data, Batch

from core.utils.model_testing import create_sequential_from_graph



class Learner(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.alignment = cfg.MODEL.ALIGNMENT
        output_dim = cfg.MODEL.OUTPUT_DIM_HEAD if cfg.MODEL.MAKE_MODEL_ENCODER_HEAD else cfg.MODEL.OUTPUT_DIM
        self.model_encoder = ModelEncoder()
        # put requres_grad to False
        #for param in self.model_encoder.parameters():
        #    param.requires_grad = False

        self.model_decoder = ModelDecoder()
        #self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        #self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        #self.sentences_encoder = SentenceTransformer('bert-base-uncased')
        # put in eval mode
        #self.sentences_encoder.eval()


        #vocab_size = self.tokenizer.vocab_size
        self.text_encoder = TextEncoder()
        #self.text_encoder = TextEncoderCustom()


        self.EdgeRecoTextLoss = nn.SmoothL1Loss()
        self.NodeRecoTextLoss = nn.SmoothL1Loss()
        self.EdgeRecoModelLoss = nn.SmoothL1Loss()
        self.NodeRecoModelLoss = nn.SmoothL1Loss()
        self.LatentLoss = nn.SmoothL1Loss()
        #self.criterion = CLIPLoss()
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
    
    def tokenize_text_batch(self, text_batch, max_length=200):
        return self.tokenizer(
            text_batch,
            padding='max_length',   # Aggiungere padding se la sequenza è più corta della lunghezza fissa
            truncation=True,        # Troncare la sequenza se è più lunga della lunghezza fissa
            max_length=max_length,  # Lunghezza fissa desiderata
            return_tensors='pt'     # Restituire tensori PyTorch
        )
    



    def forward(self, model_batch, text_batch, f=None):
        model_embed = self.model_encoder(model_batch, f)
        if self.alignment:
            text_embed = self.text_encoder(text_batch).squeeze(0)
            model_decoded_fromText = self.model_decoder(text_embed, model_batch)
        else:
            text_embed = None
            model_decoded_fromText = None
        model_decoded_fromModel = self.model_decoder(model_embed, model_batch)
        return model_embed, text_embed, model_decoded_fromModel, model_decoded_fromText

    def training_step(self, batch, batch_idx):
        # opt1 = self.optimizers()
        # opt1.zero_grad()
        # scheduler1 = self.lr_schedulers()

        model_batch, text_batch, f = batch

        # model_batch = Batch.from_data_list(model_batch)

        

        model_embed, text_embed, model_decoded_fromModel, model_decoded_fromText = self(model_batch, text_batch, f)
        # model_embed = self(model_batch, text_batch, f)
        # class_logits = self.classifier(model_embed)

        model_batch_fromModel = model_batch.clone()
        model_batch_fromModel.x = model_decoded_fromModel[0]
        model_batch_fromModel.edge_attr = model_decoded_fromModel[1]
        # RECO LOSSES
        # 1. smoothl1loss reco model from model - model gt
        node_reco_model = self.NodeRecoModelLoss(model_decoded_fromModel[0], model_batch.x)
        edge_reco_model = self.EdgeRecoModelLoss(model_decoded_fromModel[1], model_batch.edge_attr)
        reco_loss = node_reco_model + edge_reco_model

        if self.alignment:
            model_batch_fromText = model_batch.clone()
            model_batch_fromText.x = model_decoded_fromText[0]
            model_batch_fromText.edge_attr = model_decoded_fromText[1]

            # 2. smoothl1loss reco model from text - model gt
            node_reco_text = self.NodeRecoTextLoss(model_decoded_fromText[0], model_batch.x)
            edge_reco_text = self.EdgeRecoTextLoss(model_decoded_fromText[1], model_batch.edge_attr)
            reco_loss += node_reco_text + edge_reco_text

            # LATENT LOSSES
            # 3. LATENT TEXT - LATENT MODEL
            latent_loss = self.LatentLoss(model_embed, text_embed)

        
            loss = reco_loss + 1*latent_loss
        else:
            loss = reco_loss
        
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('node_reco_model', node_reco_model, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('edge_reco_model', edge_reco_model, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('reco_loss', reco_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        if self.alignment:
            self.log('node_reco_text', node_reco_text, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('edge_reco_text', edge_reco_text, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('latent_loss', latent_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # select only indices of the list f that contain "CNN2"
        #cnn2_indices = [i for i in range(len(f)) if "CNN2" in f[i]]
        #acc = self.compute_accuracy_alignment(model_embed[cnn2_indices], text_embed[cnn2_indices]
            acc = self.compute_accuracy_alignment(model_embed, text_embed)
            self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # self.manual_backward(loss)
        # opt1.step()
        # scheduler1.step()
        
        return loss

    def validation_step(self, batch, batch_idx):
        model_batch, text_batch, f = batch

        # model_batch = Batch.from_data_list(model_batch)


        model_embed, text_embed, model_decoded_fromModel, model_decoded_fromText = self(model_batch, text_batch, f)
        # model_embed = self(model_batch, text_batch, f)
        # class_logits = self.classifier(model_embed)
        
        model_batch_fromModel = model_batch.clone()
        model_batch_fromModel.x = model_decoded_fromModel[0]
        model_batch_fromModel.edge_attr = model_decoded_fromModel[1]
        # RECO LOSSES
        # 1. smoothl1loss reco model from model - model gt
        node_reco_model = self.NodeRecoModelLoss(model_decoded_fromModel[0], model_batch.x)
        edge_reco_model = self.EdgeRecoModelLoss(model_decoded_fromModel[1], model_batch.edge_attr)
        reco_loss = node_reco_model + edge_reco_model

        if self.alignment:
            model_batch_fromText = model_batch.clone()
            model_batch_fromText.x = model_decoded_fromText[0]
            model_batch_fromText.edge_attr = model_decoded_fromText[1]

            # 2. smoothl1loss reco model from text - model gt
            node_reco_text = self.NodeRecoTextLoss(model_decoded_fromText[0], model_batch.x)
            edge_reco_text = self.EdgeRecoTextLoss(model_decoded_fromText[1], model_batch.edge_attr)
            reco_loss += node_reco_text + edge_reco_text

            # LATENT LOSSES
            # 3. LATENT TEXT - LATENT MODEL
            latent_loss = self.LatentLoss(model_embed, text_embed)
            loss = reco_loss + 1*latent_loss
        else:
            loss = reco_loss

        model_orig = text_batch
        model_reco = create_sequential_from_graph(model_batch_fromModel, model_orig)

        # performances = test_on_mnist(model_reco, model_orig)


        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # select only indices of the list f that contain "CNN2"
        #cnn2_indices = [i for i in range(len(f)) if "CNN2" in f[i]]
        #acc = self.compute_accuracy_alignment(model_embed[cnn2_indices], text_embed[cnn2_indices])
        if self.alignment:
            acc = self.compute_accuracy_alignment(model_embed, text_embed)
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

        #list_parameters = list(self.model_encoder.parameters()) + list(self.text_encoder.parameters()) #+ list(self.classifier.parameters())
        #list_parameters = list(self.model_encoder.parameters()) + list(self.text_encoder_head.parameters())
        list_parameters = list(self.model_encoder.parameters()) + list(self.text_encoder.parameters())
        optimizer1 = torch.optim.AdamW(list_parameters, lr=cfg.SOLVER.BASE_LR1, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        # optimizer2 = torch.optim.AdamW(self.classifier.parameters(), lr=cfg.SOLVER.BASE_LR2, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        # scheduler1 = get_linear_schedule_with_warmup(optimizer1, num_warmup_steps=cfg.SOLVER.WARMUP_ITERS, num_training_steps=-1)
        # scheduler2 = get_linear_schedule_with_warmup(optimizer2, num_warmup_steps=cfg.SOLVER.WARMUP_ITERS, num_training_steps=-1)
        # linear_sched = LinearLR(optimizer1, start_factor=0.05, total_iters=9600)

        return [optimizer1], []