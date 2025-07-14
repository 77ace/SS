import torch
import torch.nn.functional as F
import torch.nn as nn

# Contrastive Loss Function (Normalized Temperature-scaled Cross Entropy Loss)
def nt_xent_loss(emb1, emb2, temperature=0.5):
    # Normalize embeddings to unit length (Same length)
    emb1 = F.normalize(emb1, dim=1)
    emb2 = F.normalize(emb2, dim=1)
    # Compute cosine similarity and apply temperature scaling
    logits = torch.mm(emb1, emb2.T) / temperature
    labels = torch.arange(emb1.size(0)).to(emb1.device)
    # calculate cross-entropy loss and return it
    return nn.CrossEntropyLoss()(logits, labels)
