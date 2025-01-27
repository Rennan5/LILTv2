import torch
import torch.nn as nn

class TokenVisualEmbedding(nn.Module):
    def __init__(self, embed_dim=768):
        """
        Inicializa os embeddings visuais de token.
        
        Args:
            embed_dim: Dimensão do embedding final.
        """
        super(TokenVisualEmbedding, self).__init__()
        
        # Embeddings para características visuais
        self.bold_embedding = nn.Embedding(2, embed_dim)      # 0: não é negrito, 1: é negrito
        self.italic_embedding = nn.Embedding(2, embed_dim)    # 0: não é itálico, 1: é itálico
        self.underline_embedding = nn.Embedding(2, embed_dim) # 0: não é sublinhado, 1: é sublinhado
        
        # Projeção final para combinar os embeddings
        self.projection = nn.Linear(embed_dim * 3, embed_dim)

    def forward(self, bold_flags, italic_flags, underline_flags):
        """
        Propagação para frente dos embeddings visuais de token.
        
        Args:
            bold_flags: Tensor de sinalizadores de negrito (shape: [batch_size, num_tokens]).
            italic_flags: Tensor de sinalizadores de itálico (shape: [batch_size, num_tokens]).
            underline_flags: Tensor de sinalizadores de sublinhado (shape: [batch_size, num_tokens]).
        
        Returns:
            Embeddings visuais de token (shape: [batch_size, num_tokens, embed_dim]).
        """
        # Obter embeddings para cada característica visual
        bold_embeds = self.bold_embedding(bold_flags)           # Shape: [batch_size, num_tokens, embed_dim]
        italic_embeds = self.italic_embedding(italic_flags)     # Shape: [batch_size, num_tokens, embed_dim]
        underline_embeds = self.underline_embedding(underline_flags) # Shape: [batch_size, num_tokens, embed_dim]
        
        # Concatenar os embeddings visuais
        combined_embeds = torch.cat([bold_embeds, italic_embeds, underline_embeds], dim=-1)
        
        # Projeção para reduzir a dimensionalidade
        visual_embeds = self.projection(combined_embeds)
        
        return visual_embeds