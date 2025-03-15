from transformers import BertModel
import torch
import torch.nn as nn

class MVLMTask(nn.Module):
    def __init__(self, vocab_size, embed_dim, bert_model='bert-base-uncased'):
        super(MVLMTask, self).__init__()
        self.text_embedding = nn.Embedding(vocab_size, embed_dim)
        self.layout_embedding = nn.Linear(4, embed_dim)
        self.image_embedding = nn.Linear(embed_dim, embed_dim)

        # Usando um backbone transformer pré-treinado
        self.encoder = BertModel.from_pretrained(bert_model)
        self.decoder = nn.Linear(embed_dim, vocab_size)

    def forward(self, text_tokens, layout_boxes, image_features, attention_mask):
        text_embeds = self.text_embedding(text_tokens)
        layout_embeds = self.layout_embedding(layout_boxes)
        image_embeds = self.image_embedding(image_features)

        # Concatenação + passagem pelo Transformer
        multimodal_embeds = text_embeds + layout_embeds + image_embeds
        encoded_outputs = self.encoder(inputs_embeds=multimodal_embeds, attention_mask=attention_mask)

        predictions = self.decoder(encoded_outputs.last_hidden_state)
        return predictions
class WPATask(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(WPATask, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.alignment_score = nn.Linear(embed_dim, 1)

    def forward(self, token_embeds, patch_embeds):
        attn_output, _ = self.attention(token_embeds, patch_embeds, patch_embeds)
        scores = self.alignment_score(attn_output).squeeze(-1)
        return scores
    
class KPLTask(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(KPLTask, self).__init__()
        self.spatial_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads),
            num_layers=3
        )
        self.mlp = nn.Linear(embed_dim, 4)

    def forward(self, text_embeds):
        encoded = self.spatial_transformer(text_embeds)
        return self.mlp(encoded)
    
class RPCTask(nn.Module):
    def __init__(self, embed_dim, num_classes, num_heads=8):
        super(RPCTask, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim * 2, nhead=num_heads),
            num_layers=2
        )
        self.mlp = nn.Linear(embed_dim * 2, num_classes)

    def forward(self, box1_embeds, box2_embeds):
        relative_features = torch.cat([box1_embeds, box2_embeds], dim=-1)
        encoded_features = self.encoder(relative_features)
        return self.mlp(encoded_features)
