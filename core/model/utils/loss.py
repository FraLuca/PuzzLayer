import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPLoss(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super(CLIPLoss, self).__init__()
        self.temperature = temperature

    def forward(self, image_features, text_features):
        # Normalizza le rappresentazioni delle immagini e dei testi
        image_features = F.normalize(image_features, dim=1)
        text_features = F.normalize(text_features, dim=1)

        # Calcola i logit
        logits_per_image = image_features @ text_features.t() / self.temperature
        logits_per_text = text_features @ image_features.t() / self.temperature

        # Crea etichette target (0, 1, 2, ..., batch_size-1)
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)

        # Calcola la perdita
        loss_img_to_txt = F.cross_entropy(logits_per_image, labels)
        loss_txt_to_img = F.cross_entropy(logits_per_text, labels)
        
        # La perdita totale è la media delle due direzioni
        loss = (loss_img_to_txt + loss_txt_to_img) / 2
        return loss