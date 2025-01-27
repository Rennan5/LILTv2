import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        """
        LoRA layer for adapting a dense layer.

        Args:
            in_features (int): Input dimensionality.
            out_features (int): Output dimensionality.
            rank (int): Low-rank value.
            alpha (float): Scaling factor for LoRA updates.
        """
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.alpha = alpha

        self.weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)

        self.A = nn.Parameter(torch.randn(out_features, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, in_features) * 0.01)

        self.scaling = alpha / rank

    def forward(self, x):
        """
        Forward pass for the LoRA layer.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_features).
        """
        delta_w = torch.einsum("oi,ij->oj", self.A, self.B) * self.scaling

        adapted_weight = self.weight + delta_w

        return x @ adapted_weight.T