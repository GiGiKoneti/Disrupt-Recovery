"""
PyTorch Contrastive Encoder Network

Architecture:
- Input: concatenated multi-level feature vector
- Multi-layer MLP with batch norm and dropout
- Output: 256-dim normalised embedding

Trained with supervised contrastive loss to separate:
- Human vs AI text
- Different model families (GPT, Claude, Gemini, etc.)

Author: SynthDetect Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveEncoder(nn.Module):
    """
    MLP-based contrastive encoder for stylometric features.

    Maps high-dimensional feature vectors to a compact embedding
    space where texts from the same source cluster together.
    """

    def __init__(
        self,
        input_dim: int = 195,   # sum of all feature dimensions
        hidden_dims: list = None,
        embedding_dim: int = 256,
        dropout: float = 0.3,
    ):
        """
        Args:
            input_dim: Dimension of input feature vector.
            hidden_dims: List of hidden layer sizes.
            embedding_dim: Output embedding dimension.
            dropout: Dropout probability.
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256]

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        # Projection head
        layers.append(nn.Linear(prev_dim, embedding_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input feature tensor [batch_size, input_dim].

        Returns:
            L2-normalised embedding [batch_size, embedding_dim].
        """
        embedding = self.encoder(x)
        # L2 normalise for cosine similarity in contrastive loss
        return F.normalize(embedding, p=2, dim=1)


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.

    Pulls embeddings of same-class samples together
    and pushes different-class samples apart.

    Reference: Khosla et al. (2020) "Supervised Contrastive Learning"
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute supervised contrastive loss.

        Args:
            features: L2-normalised embeddings [batch_size, embedding_dim].
            labels: Class labels [batch_size].

        Returns:
            Scalar loss value.
        """
        device = features.device
        batch_size = features.shape[0]

        # Compute similarity matrix
        similarity = torch.matmul(features, features.T) / self.temperature

        # Create mask for positive pairs (same label)
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Remove self-contrast
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        # Compute log softmax
        exp_logits = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Compute mean of log-likelihood over positive pairs
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.clamp(mask_sum, min=1)  # Avoid division by zero
        mean_log_prob = (mask * log_prob).sum(dim=1) / mask_sum

        # Loss
        loss = -mean_log_prob.mean()
        return loss
