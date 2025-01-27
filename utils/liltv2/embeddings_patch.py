import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        """
        Inicializa os embeddings em nível de patch.
        
        Args:
            img_size: Tamanho da imagem (assumindo quadrada).
            patch_size: Tamanho de cada patch.
            in_channels: Número de canais na imagem (e.g., 3 para RGB).
            embed_dim: Dimensão do embedding projetado.
        """
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Projeção linear dos patches
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Embeddings de posição
        self.position_embeddings = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
    
    def forward(self, x):
        """
        Propagação para frente.
        
        Args:
            x: Entrada da imagem (shape: [batch_size, in_channels, img_size, img_size]).
            
        Returns:
            Embeddings de patches (shape: [batch_size, num_patches, embed_dim]).
        """
        batch_size = x.size(0)
        
        # Projeção para embeddings
        x = self.projection(x)  # Shape: [batch_size, embed_dim, H_patches, W_patches]
        x = x.flatten(2).transpose(1, 2)  # Shape: [batch_size, num_patches, embed_dim]
        
        # Adicionar embeddings de posição
        x = x + self.position_embeddings
        
        return x