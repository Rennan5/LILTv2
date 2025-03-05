import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha / rank
        self.A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, out_features) * 0.01)

    def forward(self, x, weight):
        delta_w = self.A @ self.B * self.alpha
        return x @ (weight + delta_w).T

class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=4, alpha=1.0):
        """
        Substitui uma camada Linear por uma versão compatível com LoRA, mantendo a camada original.

        Args:
            original_linear (nn.Linear): Camada Linear original a ser ajustada com LoRA.
            rank (int): Dimensão do espaço de menor dimensão para LoRA.
            alpha (float): Fator de escala para a adaptação.
        """
        super().__init__()
        self.original_linear = original_linear
        self.lora = LoRALayer(original_linear.in_features, original_linear.out_features, rank, alpha)

    def forward(self, x):
        """
        Forward passa o input tanto pela camada original quanto pela LoRA.

        Args:
            x (torch.Tensor): Entrada do modelo.

        Returns:
            torch.Tensor: Saída combinada da camada original e da LoRA.
        """
        return self.original_linear(x) + self.lora(x)
