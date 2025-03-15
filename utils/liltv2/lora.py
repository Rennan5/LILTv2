import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super(LoRALayer, self).__init__()
        
        self.rank = rank
        self.alpha = alpha

        self.weight = nn.Parameter(torch.randn(out_features, in_features, device="cuda"), requires_grad=True)
        self.A = nn.Parameter(torch.randn(out_features, rank, device="cuda") * 0.001)
        self.B = nn.Parameter(torch.randn(rank, in_features, device="cuda") * 0.001)

        self.scaling = alpha / rank

    def forward(self, x):

        weight = self.weight.to(x.device)
        delta_w = (self.A.to(x.device) @ self.B.to(x.device) * self.scaling).to(x.device)

        x = x.to(torch.float32)
        weight = weight.to(torch.float32)
        delta_w = delta_w.to(torch.float32)

        return x @ (weight + delta_w).T

class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=4, alpha=1.0):
        super().__init__()
        self.original_linear = original_linear
        self.lora = LoRALayer(original_linear.in_features, original_linear.out_features, rank, alpha)

    def forward(self, x):
        """
        Passa o input pela camada original e pela adaptação LoRA.
        """
        return self.original_linear(x) + self.lora(x)

