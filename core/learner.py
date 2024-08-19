import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from core.model.build import build_model, init_model
from core.model.denoiser import Denoiser
from core.model.text_encoder import TextEncoder
from core.dataset.build import build_dataset, custom_collate_fn
from core.model.utils.loss import CLIPLoss
from core.configs import cfg
import numpy as np
import diffusers
from torch_geometric.data import Data, Batch

from torch_geometric.data import Batch
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import LinearLR
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from core.model.utils.metrics import *
from core.model.head import ProjectionHead
from core.utils.model_testing import *


class Learner(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # used at test time
        self.ddim_scheduler = diffusers.DDIMScheduler(
                num_train_timesteps= 1000,
                beta_start= 0.00085,
                beta_end= 0.012,
                beta_schedule= 'scaled_linear', # Optional: ['linear', 'scaled_linear', 'squaredcos_cap_v2']
                clip_sample= False,
                set_alpha_to_one= False,
                steps_offset= 1
        )
        self.ddim_scheduler.set_timesteps(50)
        self.eta_ddim = 0.0
        self.do_classifier_free_guidance = True
        self.guidance_scale = 7.5
        self.cfg_prob = 0.1
        
        # used at train time
        self.ddpm_scheduler = diffusers.DDPMScheduler(
                num_train_timesteps= 1000,
                beta_start= 0.00085,
                beta_end= 0.012,
                beta_schedule= 'scaled_linear', # Optional: ['linear', 'scaled_linear', 'squaredcos_cap_v2']
                variance_type= 'fixed_small',
                clip_sample= False,
        )

        self.model_denoiser = Denoiser()

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

        self.criterion = nn.MSELoss()

        if cfg.PRETRAINED_MODEL_ENCODER:
            print(f"Loading pretrained model encoder from {cfg.PRETRAINED_MODEL_ENCODER}")
            self.load_checkpoint(cfg.PRETRAINED_MODEL_ENCODER)

        self.save_hyperparameters(cfg)


    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        pre_weights = {k: v for k, v in checkpoint["state_dict"].items() if "model_encoder" in k}
        self.load_state_dict(pre_weights, strict=False)


    def forward(self, model_batch, text_batch, f=None):
        # model_embed = self.model_projection(self.model_encoder(model_batch, f))
        text_embed = self.text_projection(self.text_encoder(text_batch))
        return model_embed, text_embed

    def training_step(self, batch, batch_idx):
        model_batch, text_batch, f, sequential, layers_mask = batch

        # model_embed, text_embed = self(model_batch, text_batch, f)
        noise_set = self.train_diffusion_forward(batch)
        loss = self.criterion(noise_set["noise_pred"], noise_set["noise"])
        # else:
        #     loss += self.criterion(noise_set["noise_pred"], noise_set["orig"])

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        model_batch, text_batch, f, sequential, layers_mask = batch

        generated = self.test_diffusion(batch)
        loss = self.criterion(generated.edge_attr[:,0:1], model_batch.edge_attr[:,0:1])

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # sim = self.compute_sim_matrix(model_embed, text_embed, f)
        # # acc = self.compute_accuracy_alignment(model_embed, text_embed, f)
        # # self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # for k in [1, 3]:
        #     recall_i2t, recall_t2i = recall_at_k(sim, k)
        #     self.log(f"val_recall_i2t@{k}", recall_i2t, on_step=False, on_epoch=True, sync_dist=True)
        #     self.log(f"val_recall_t2i@{k}", recall_t2i, on_step=False, on_epoch=True, sync_dist=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        model_batch, text_batch, f, sequential, layers_mask = batch

        generated = self.test_diffusion(batch)
        loss = self.criterion(generated.edge_attr[:,0:1], model_batch.edge_attr[:,0:1])

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # evaluate generated models on MNIST
        new_sequential = create_sequentials_from_graphs(generated, sequential)
        avg_reco_acc, avg_orig_acc = test_on_mnist(new_sequential, sequential, f)

        self.log('test_reco_acc', avg_reco_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('test_orig_acc', avg_orig_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # sim = self.compute_sim_matrix(model_embed, text_embed, f)
        # # acc = self.compute_accuracy_alignment(model_embed, text_embed, f)
        # # self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # for k in [1, 3]:
        #     recall_i2t, recall_t2i = recall_at_k(sim, k)
        #     self.log(f"test_recall_i2t@{k}", recall_i2t, on_step=False, on_epoch=True, sync_dist=True)
        #     self.log(f"test_recall_t2i@{k}", recall_t2i, on_step=False, on_epoch=True, sync_dist=True)

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
            pin_memory=False,
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
            {"params": self.model_denoiser.parameters(), "lr": self.cfg.SOLVER.MODEL_ENCODER_LR},
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
    
    ###############################################################
    # diffusion things

    def train_diffusion_forward(self, batch):
        model_batch, text_batch, f, sequential, layers_mask = batch

        if self.do_classifier_free_guidance:
            # classifier free guidance: randomly drop text during training
            text_batch = [
                "" if np.random.rand(1) < self.cfg_prob else i
                for i in text_batch
            ]

        text = self.tokenizer(
                text_batch,
                padding=True,
                return_tensors="pt",
            ).to(model_batch.x.device).input_ids # TODO: remember that we are not removing the first and last token, and we are not using attention_masks
        # text encode
        text_emb = self.text_projection(self.text_encoder(text))

        # diffusion process return with noise and noise_pred
        n_set = self._diffusion_process(model_batch, text_emb, layers_mask)
        return {**n_set}
    
    def _diffusion_process(self, models, text_emb, layers_mask, lengths=None):
        """
        heavily from https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py
        """
        # our latent   [batch_size, n_token=1 or 5 or 10, latent_dim=256]
        # sd  latent   [batch_size, [n_token0=64,n_token1=64], latent_dim=4]
        # [n_token, batch_size, latent_dim] -> [batch_size, n_token, latent_dim]
        # latents = latents.permute(1, 0, 2)
        
        # orig_models = models.clone()

        # Sample noise that we'll add to the latents
        # [batch_size, n_token, latent_dim]
        noise = torch.randn_like(models.edge_attr[:,0:1])
        if cfg.MODEL.DIFFUSION_PER_LAYER:
            # put non-zero noise only for the layers that we want to diffuse
            noise = noise * layers_mask

        edge_batch = models.batch[models.edge_index[0]] # edge_batch will contain the graph index for each edge
        
        n_graphs = models.batch[-1] + 1
        # Sample a random timestep for each graph in the batch
        timesteps = torch.randint(
            0,
            1000, #self.noise_scheduler.config.num_train_timesteps,
            (n_graphs, ),
            device=models.x.device,
        )

        # now repeat same timestep for all edges in the same graph
        timesteps = timesteps[edge_batch]
        # now repeat same text_emb for all edges in the same graph
        text_emb = text_emb[edge_batch]

        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_edge_attr = self.ddpm_scheduler.add_noise(models.edge_attr[:,0:1].clone(), noise, timesteps)

        # put back noised weights inside batch
        if cfg.MODEL.DIFFUSION_PER_LAYER:
            # fill the noised edge_attr only for the layers that we want to diffuse
            models.edge_attr[:,0:1] = models.edge_attr[:,0:1] * (1 - layers_mask) + noisy_edge_attr * layers_mask
        else:
            models.edge_attr[:,0:1] = noisy_edge_attr

        # Predict the noise residual
        noise_pred = self.model_denoiser(
            sample=models,
            timestep=timesteps,
            text_emb=text_emb,
            layers_mask=layers_mask,
            return_dict=False,
        )[0]

        if cfg.MODEL.DIFFUSION_PER_LAYER:
            # keep the noise_pred only for the layers that we diffused
            noise_pred = noise_pred * layers_mask

        noise_pred_prior = 0
        noise_prior = 0
        n_set = {
            "noise": noise,
            "noise_prior": noise_prior,
            "noise_pred": noise_pred,
            "noise_pred_prior": noise_pred_prior,
        }
        # if not self.predict_epsilon:
        #     n_set["pred"] = noise_pred
        #     n_set["orig"] = orig_models.edge_attr[:,0:1]
        return n_set
    
    def test_diffusion(self, batch):
        model_batch, text_batch, f, sequential, layers_mask = batch

        edge_batch = model_batch.batch[model_batch.edge_index[0]] # edge_batch will contain the graph index for each edge

        text_batch_tokenized = self.tokenizer(
                text_batch,
                padding=True,
                return_tensors="pt",
            ).to(model_batch.x.device).input_ids # TODO: remember that we are not removing the first and last token, and we are not using attention_masks
        text_batch_embedded = self.text_projection(self.text_encoder(text_batch_tokenized))
        # now repeat same text_emb for all edges in the same graph
        text_batch_embedded = text_batch_embedded[edge_batch]

        if self.do_classifier_free_guidance:
            # cfg
            uncond_tokens = [""] * len(text_batch)
            uncond_tokens = self.tokenizer(
                            uncond_tokens,
                            padding="max_length",
                            truncation=True,
                            max_length=6 if cfg.MODEL.DIFFUSION_PER_LAYER else 5,
                            return_tensors="pt",
                        ).to(model_batch.x.device).input_ids
            uncond_tokens = self.text_projection(self.text_encoder(uncond_tokens))
            uncond_tokens = uncond_tokens[edge_batch]
            text_emb = torch.cat([uncond_tokens, text_batch_embedded], dim=0) # unconditioned first, then conditioned
        else:
            text_emb = text_batch_embedded
        
        with torch.no_grad():
            generated = self._diffusion_reverse(model_batch, text_emb, layers_mask)

        return generated

    def _diffusion_reverse(self, model_batch, text_emb, layers_mask):
        # init latents
        bsz = text_emb.shape[0] # bsz will correspond to number of edges
        if self.do_classifier_free_guidance:
            bsz = bsz // 2
            layers_mask_cfg = None
            if cfg.MODEL.DIFFUSION_PER_LAYER:
                # concatenate the layers mask for the unconditioned and conditioned predictions
                layers_mask_cfg = torch.cat([layers_mask, layers_mask], dim=0)
        
        model_batch_orig = model_batch.clone()
        
        noised_weights = torch.randn(
            (bsz, 1),
            device=model_batch.edge_attr.device,
            dtype=torch.float,
        )

        # scale the initial noise by the standard deviation required by the scheduler
        noised_weights = noised_weights * self.ddim_scheduler.init_noise_sigma

        # set timesteps
        # self.ddim_scheduler.set_timesteps(50)
        timesteps = self.ddim_scheduler.timesteps.to(model_batch.edge_attr.device)
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, and between [0, 1]
        extra_step_kwargs = {}
        if "eta" in set(
                inspect.signature(self.ddim_scheduler.step).parameters.keys()):
            extra_step_kwargs["eta"] = self.eta_ddim

        # reverse
        for i, t in enumerate(timesteps):
            if cfg.MODEL.DIFFUSION_PER_LAYER:
                # fill the noised weights only for the layers that we want to diffuse
                model_batch_orig.edge_attr[:,0:1] = model_batch_orig.edge_attr[:,0:1] * (1 - layers_mask) + noised_weights * layers_mask
            else:
                model_batch_orig.edge_attr[:,0:1] = noised_weights
            if self.do_classifier_free_guidance:
                # expand the input if we are doing cfg: replicate model_batch two times
                model_batch = Batch.from_data_list([model_batch_orig, model_batch_orig.clone()])
            else:
                model_batch = model_batch_orig
            
            # now repeat same timestep for all edges
            t_batched = t.expand(model_batch.edge_attr.shape[0])
            
            # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # predict the noise residual
            noise_pred = self.model_denoiser(
                sample=model_batch,
                timestep=t_batched,
                text_emb=text_emb,
                layers_mask=layers_mask_cfg if self.do_classifier_free_guidance else layers_mask,
            )[0]
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond)

            noised_weights = self.ddim_scheduler.step(noise_pred, t, noised_weights,
                                              **extra_step_kwargs).prev_sample
            # if self.predict_epsilon:
            #     latents = self.scheduler.step(noise_pred, t, latents,
            #                                   **extra_step_kwargs).prev_sample
            # else:
            #     # predict x for standard diffusion model
            #     # compute the previous noisy sample x_t -> x_t-1
            #     latents = self.scheduler.step(noise_pred,
            #                                   t,
            #                                   latents,
            #                                   **extra_step_kwargs).prev_sample
        
        if cfg.MODEL.DIFFUSION_PER_LAYER:
            # fill the noised weights only for the layers that we want to diffuse
            model_batch_orig.edge_attr[:,0:1] = model_batch_orig.edge_attr[:,0:1] * (1 - layers_mask) + noised_weights * layers_mask
        else:
            model_batch_orig.edge_attr[:,0:1] = noised_weights
        return model_batch_orig