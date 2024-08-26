import torch

def recall_at_k(sim, k):
    """
    Compute Recall@K for both Image-to-Text (I2T) and Text-to-Image (T2I) retrieval.
    
    Args:
        sim (torch.Tensor): Similarity matrix of shape (num_images, num_texts).
        k (int): The value of K for Recall@K.
        
    Returns:
        recall_i2t (float): Recall@K for Image-to-Text retrieval.
        recall_t2i (float): Recall@K for Text-to-Image retrieval.
    """
    num_images, num_texts = sim.size()

    # Image-to-Text (I2T) Retrieval
    topk_texts = sim.topk(k, dim=1).indices  # Get top K indices along the text dimension for each image
    correct_texts = torch.arange(num_images).unsqueeze(1).expand_as(topk_texts).to(topk_texts.device)
    recall_i2t = (topk_texts == correct_texts).any(dim=1).float().mean().item()

    # Text-to-Image (T2I) Retrieval
    topk_images = sim.topk(k, dim=0).indices  # Get top K indices along the image dimension for each text
    correct_images = torch.arange(num_texts).unsqueeze(0).expand_as(topk_images).to(topk_images.device)
    recall_t2i = (topk_images == correct_images).any(dim=0).float().mean().item()

    return recall_i2t, recall_t2i