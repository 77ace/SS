import torch
import torch.nn.functional as F

def sup_contrastive_loss(emb1, emb2, labels, temperature=0.1):
    """
    Supervised Contrastive Loss Function
    Args:
        emb1: Embeddings from the first input (shape: [batch_size, embedding_dim])
        emb2: Embeddings from the second input (shape: [batch_size, embedding_dim])
        labels: Ground truth labels for the inputs (shape: [batch_size])
        temperature: Temperature scaling factor
    Returns:
        loss: Computed supervised contrastive loss
    """
    # Normalize embeddings
    emb1 = F.normalize(emb1, dim=1)
    emb2 = F.normalize(emb2, dim=1)

    # Combine them into one stack
    embeddings = torch.cat([emb1, emb2], dim=0)  # [2B, D]
    labels = torch.cat([labels, labels], dim=0)  # [2B]

    # Compute similarity matrix
    sim_matrix = torch.mm(embeddings, embeddings.T) / temperature

    # Mask out self-comparisons
    self_mask = torch.eye(sim_matrix.size(0), device=embeddings.device).bool()
    sim_matrix = sim_matrix.masked_fill(self_mask, -9e15)

    # Create label match mask
    label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)

    # For each anchor, compute log probability of matching pairs
    numerator = torch.exp(sim_matrix) * label_mask
    denominator = torch.exp(sim_matrix).sum(dim=1, keepdim=True)

    # Compute contrastive loss
    loss = -torch.log(numerator.sum(dim=1) / denominator.squeeze())
    loss = loss.mean()
    return loss