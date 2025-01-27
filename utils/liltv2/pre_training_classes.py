import torch
import torch.nn as nn

class MVLMTask(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        """
        Implementa a tarefa MVLM (Masked Visual-Language Modeling).
        
        Args:
            vocab_size: Tamanho do vocabulário.
            embed_dim: Dimensão dos embeddings.
        """
        super(MVLMTask, self).__init__()
        self.text_embedding = nn.Embedding(vocab_size, embed_dim)
        self.layout_embedding = nn.Linear(4, embed_dim)  # Coordenadas (x1, y1, x2, y2)
        self.image_embedding = nn.Linear(embed_dim, embed_dim)
        self.decoder = nn.Linear(embed_dim, vocab_size)

    def forward(self, text_tokens, layout_boxes, image_features, masked_indices):
        """
        Args:
            text_tokens: Tensor de texto (B, T) com IDs dos tokens.
            layout_boxes: Tensor com bounding boxes (B, T, 4).
            image_features: Tensor de características visuais (B, T, embed_dim).
            masked_indices: Máscara indicando os tokens mascarados (B, T).
        
        Returns:
            Previsões de tokens mascarados (B, T, vocab_size).
        """
        # Embeddings
        text_embeds = self.text_embedding(text_tokens)          # (B, T, embed_dim)
        layout_embeds = self.layout_embedding(layout_boxes)     # (B, T, embed_dim)
        image_embeds = self.image_embedding(image_features)     # (B, T, embed_dim)

        # Combinação multimodal
        combined_embeds = text_embeds + layout_embeds + image_embeds

        # Aplicar a máscara
        masked_embeds = combined_embeds[masked_indices]

        # Previsão dos tokens mascarados
        predictions = self.decoder(masked_embeds)  # (num_masked, vocab_size)
        return predictions

class WPATask(nn.Module):
    def __init__(self, embed_dim):
        super(WPATask, self).__init__()
        self.alignment_score = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),  # Escalar de alinhamento
            nn.Sigmoid()  # Probabilidade de alinhamento
        )

    def forward(self, token_embeds, patch_embeds):
        """
        Args:
            token_embeds: Embeddings dos tokens de texto (B, T, embed_dim).
            patch_embeds: Embeddings dos patches de imagem (B, P, embed_dim).
        
        Returns:
            Probabilidades de alinhamento (B, T, P).
        """
        # Expandir dimensões para calcular alinhamento entre todos os pares
        T, P = token_embeds.size(1), patch_embeds.size(1)
        token_embeds = token_embeds.unsqueeze(2).expand(-1, -1, P, -1)  # (B, T, P, embed_dim)
        patch_embeds = patch_embeds.unsqueeze(1).expand(-1, T, -1, -1)  # (B, T, P, embed_dim)

        # Concatenar e calcular score de alinhamento
        pairwise_features = torch.cat([token_embeds, patch_embeds], dim=-1)  # (B, T, P, 2*embed_dim)
        alignment_scores = self.alignment_score(pairwise_features).squeeze(-1)  # (B, T, P)
        return alignment_scores
    
class KPLTask(nn.Module):
    def __init__(self, embed_dim):
        """
        Implementa a tarefa KPL (Key Point Location).
        
        Args:
            embed_dim: Dimensão dos embeddings.
        """
        super(KPLTask, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 4)  # Previsão de (x1, y1, x2, y2)
        )

    def forward(self, text_embeds):
        """
        Args:
            text_embeds: Tensor de embeddings de texto (B, T, embed_dim).
        
        Returns:
            Previsão de bounding boxes (B, T, 4).
        """
        return self.mlp(text_embeds)

class RPCTask(nn.Module):
    def __init__(self, embed_dim, num_classes):
        """
        Implementa a tarefa RPC (Relative Position Classification).
        
        Args:
            embed_dim: Dimensão dos embeddings.
            num_classes: Número de classes para posição relativa.
        """
        super(RPCTask, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_classes)  # Classes para posição relativa
        )

    def forward(self, box1_embeds, box2_embeds):
        """
        Args:
            box1_embeds: Embeddings da primeira caixa (B, T, embed_dim).
            box2_embeds: Embeddings da segunda caixa (B, T, embed_dim).
        
        Returns:
            Classes previstas para a posição relativa (B, T, num_classes).
        """
        relative_features = torch.cat([box1_embeds, box2_embeds], dim=-1)
        return self.mlp(relative_features)