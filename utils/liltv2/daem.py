import torch
import torch.nn as nn

class DAEM(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, n_layers, dropout=0.1):
        super(DAEM, self).__init__()
        self.layers = nn.ModuleList([
            DAEMLayer(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, text_embeds, image_embeds, text_layout, image_layout):
        """
        Forward pass through DAEM layers.

        Args:
            text_embeds: Text embeddings (B, T, d_model)
            image_embeds: Image embeddings (B, I, d_model)
            text_layout: Text layout embeddings (B, T, d_model)
            image_layout: Image layout embeddings (B, I, d_model)

        Returns:
            Updated text and image embeddings.
        """
        # Combine embeddings with layout information
        text_input = text_embeds + text_layout
        image_input = image_embeds + image_layout

        for layer in self.layers:
            text_input, image_input = layer(text_input, image_input)

        return text_input, image_input


class DAEMLayer(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1):
        super(DAEMLayer, self).__init__()
        # Cross-attention layers
        self.text_to_image_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.image_to_text_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward networks
        self.text_ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.image_ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

        # Layer norms
        self.text_norm1 = nn.LayerNorm(d_model)
        self.text_norm2 = nn.LayerNorm(d_model)
        self.image_norm1 = nn.LayerNorm(d_model)
        self.image_norm2 = nn.LayerNorm(d_model)

    def forward(self, text_embeds, image_embeds):
        """
        Forward pass through one DAEM layer.

        Args:
            text_embeds: Text embeddings (B, T, d_model)
            image_embeds: Image embeddings (B, I, d_model)

        Returns:
            Updated text and image embeddings.
        """
        text_to_image_out, _ = self.text_to_image_attn(
            query=text_embeds, key=image_embeds, value=image_embeds
        )
        text_out = self.text_norm1(text_embeds + text_to_image_out)

        image_to_text_out, _ = self.image_to_text_attn(
            query=image_embeds, key=text_embeds, value=text_embeds
        )
        image_out = self.image_norm1(image_embeds + image_to_text_out)

        text_out = self.text_norm2(text_out + self.text_ffn(text_out))
        image_out = self.image_norm2(image_out + self.image_ffn(image_out))

        return text_out, image_out
