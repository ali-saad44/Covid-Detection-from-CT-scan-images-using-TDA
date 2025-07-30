from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy
import torch
import numpy as np

def extract_tda_features(images):
    """
    Extract persistent homology features from a batch of images using giotto-tda.
    Args:
        images (torch.Tensor): Batch of images (batch, channels, H, W)
    Returns:
        torch.Tensor: TDA features (batch, feature_dim)
    """
    images_np = images.cpu().numpy()
    if images_np.shape[1] == 3:
        images_np = np.mean(images_np, axis=1)  # Convert to grayscale
    diagrams = VietorisRipsPersistence(homology_dimensions=[0, 1]).fit_transform(images_np)
    entropy = PersistenceEntropy().fit_transform(diagrams)
    return torch.tensor(entropy, dtype=torch.float32)
