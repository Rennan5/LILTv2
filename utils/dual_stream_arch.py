import torch
import torch.nn as nn
from transformers import AutoModel

class DualStreamAttentionLiLTv2(nn.Module):
    def __init__(self, base_model_name="bert-base-uncased", num_tasks=3, task_heads=None):
        super(DualStreamAttentionLiLTv2, self).__init__()

        # Encoder para o texto
        self.text_encoder = AutoModel.from_pretrained(base_model_name)

        # Encoder para imagem/layout (substitui a MLP)
        self.layout_encoder = nn.Conv2d(3, self.text_encoder.config.hidden_size, kernel_size=3, stride=1, padding=1)

        # Dual-stream Attention Enhancement Mechanism (DAEM)
        self.text_to_layout_attn = nn.MultiheadAttention(
            embed_dim=self.text_encoder.config.hidden_size, num_heads=8, batch_first=True
        )
        self.layout_to_text_attn = nn.MultiheadAttention(
            embed_dim=self.text_encoder.config.hidden_size, num_heads=8, batch_first=True
        )

        # Fusão aprimorada
        self.fusion_layer = nn.Linear(2 * self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size)

        # Heads para tarefas específicas
        self.task_heads = nn.ModuleList()
        for task in task_heads:
            if task['type'] == 'classification':
                head = nn.Sequential(nn.Linear(self.text_encoder.config.hidden_size, task['output_size']), nn.Softmax(dim=-1))
            elif task['type'] == 'regression':
                head = nn.Linear(self.text_encoder.config.hidden_size, task['output_size'])
            else:
                raise ValueError(f"Unsupported task type: {task['type']}")
            self.task_heads.append(head)

    def forward(self, input_ids, attention_mask, layout_images, task_id):
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_hidden_state = text_outputs.last_hidden_state  # Shape: (B, T, D)

        # Layout embeddings (CNN-based)
        layout_embeds = self.layout_encoder(layout_images)  # Shape: (B, D, H, W)
        layout_embeds = layout_embeds.flatten(2).transpose(1, 2)  # Shape: (B, L, D)

        # Aplicar DAEM (atenção cruzada)
        text_refined, _ = self.text_to_layout_attn(text_hidden_state, layout_embeds, layout_embeds)
        layout_refined, _ = self.layout_to_text_attn(layout_embeds, text_hidden_state, text_hidden_state)

        # Combinar os embeddings refinados
        combined_state = torch.cat((text_refined[:, 0], layout_refined[:, 0]), dim=-1)
        fused_state = self.fusion_layer(combined_state)

        # Passar para a task head
        output = self.task_heads[task_id](fused_state)
        return output