import torch
import torch.nn as nn
from transformers import AutoModel

class DualStreamAttentionLiLTv2(nn.Module):
    def __init__(self, base_model_name="bert-base-uncased", tokenizer=None, num_tasks=3, task_heads=None):
        super(DualStreamAttentionLiLTv2, self).__init__()

        self.tokenizer = tokenizer  # Opcional para evitar erro

        # Encoder para o fluxo de texto
        self.text_encoder = AutoModel.from_pretrained("SCUT-DLVCLab/lilt-roberta-en-base")
        hidden_dim = self.text_encoder.config.hidden_size  # 768

        # Ajuste correto da camada de projeção do layout
        self.layout_encoder = nn.Linear(hidden_dim, hidden_dim)

        # Dual-stream Attention Enhancement Mechanism (DAEM)
        self.text_to_layout_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        self.layout_to_text_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)

        # Aprimoramento com viés espacial Û
        self.relative_position_bias = nn.Linear(4, hidden_dim)

        # Camada de fusão aprimorada
        self.fusion_layer = nn.Linear(2 * hidden_dim, hidden_dim)

        # Heads para tarefas específicas
        self.task_heads = nn.ModuleList()
        for task in task_heads:
            if task['type'] == 'classification':
                head = nn.Sequential(nn.Linear(hidden_dim, task['output_size']), nn.Softmax(dim=-1))
            elif task['type'] == 'regression':
                head = nn.Linear(hidden_dim, task['output_size'])
            else:
                raise ValueError(f"Unsupported task type: {task['type']}")
            self.task_heads.append(head)

    def forward(self, input_ids, attention_mask, layout_images, text_positions, image_positions, task_id):
        
        # Saída do encoder de texto
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_hidden_state = text_outputs.last_hidden_state  # (B, T, D)

        # Ajuste de layout_images antes da projeção
        if layout_images.dim() == 4:  # Caso sejam imagens (B, C, H, W)
            layout_images = layout_images.flatten(2).transpose(1, 2)  # (B, L, D)

        # Garantir que layout_images tenha shape correto antes da projeção
        hidden_dim = self.text_encoder.config.hidden_size
        if layout_images.shape[-1] != hidden_dim:
            layout_images = torch.nn.functional.pad(layout_images, (0, hidden_dim - layout_images.shape[-1]))
        
        if layout_images.shape[0] != text_hidden_state.shape[0]:
            layout_images = layout_images.expand(text_hidden_state.shape[0], -1, -1)

        layout_embeds = self.layout_encoder(layout_images)  # Projeta para `hidden_dim`

        # Cálculo do viés de posição relativa Û
        delta_positions = self.compute_relative_positions(text_positions, image_positions)
        position_bias = self.relative_position_bias(delta_positions)  # (B, T, L, hidden_dim)
    
        seq_len_text = text_hidden_state.shape[1]  # 512
        seq_len_layout = layout_embeds.shape[1]    # 672

        if seq_len_layout > seq_len_text:
            layout_embeds = layout_embeds[:, :seq_len_text, :]  # Trunca para 512
        elif seq_len_layout < seq_len_text:
            pad_size = seq_len_text - seq_len_layout
            padding = torch.zeros(layout_embeds.shape[0], pad_size, layout_embeds.shape[-1], device=layout_embeds.device)
            layout_embeds = torch.cat([layout_embeds, padding], dim=1)

        text_hidden_state = text_hidden_state.transpose(0, 1)  # (T, B, D)
        layout_embeds = layout_embeds.transpose(0, 1)          # (L, B, D)

        # Aplicação da atenção cruzada (DAEM)
        text_refined, _ = self.text_to_layout_attn(text_hidden_state, layout_embeds, layout_embeds)
        layout_refined, _ = self.layout_to_text_attn(layout_embeds, text_hidden_state, text_hidden_state)
        
        position_bias = position_bias.mean(dim=2)  # Faz média ao longo de `L`
        
        position_bias = position_bias.permute(1, 2, 0)  # (B, hidden_dim, T_cur)
        position_bias = torch.nn.functional.interpolate(position_bias, size=text_refined.shape[0], mode="linear", align_corners=True)
        position_bias = position_bias.permute(2, 0, 1)  # (T, B, hidden_dim)

        text_refined += position_bias
        layout_refined += position_bias

        # Combinar embeddings refinados
        combined_state = torch.cat((text_refined[:, 0], layout_refined[:, 0]), dim=-1)
        fused_state = self.fusion_layer(combined_state)

        # Passar para a task head correspondente
        output = self.task_heads[task_id](fused_state)
        return output

    def compute_relative_positions(self, text_positions, image_positions):
        """
        Calcula a matriz de diferenças posicionais conforme descrito no artigo
        """
        eps = 1e-6  # Pequeno valor para evitar log(0)

        # Calcula as diferenças entre coordenadas de texto e imagem
        delta_x = text_positions[:, None, 0] - image_positions[:, 0]  # (B, T, L)
        delta_y = text_positions[:, None, 1] - image_positions[:, 1]
        delta_w = torch.log((text_positions[:, None, 2] + eps) / (image_positions[:, 2] + eps))
        delta_h = torch.log((text_positions[:, None, 3] + eps) / (image_positions[:, 3] + eps))

        # Stack final esperado: (B, T, L, 4)
        relative_positions = torch.stack([delta_x, delta_y, delta_w, delta_h], dim=-1)

        if relative_positions.shape[2] == 1:
            relative_positions = relative_positions.squeeze(2)  # Remove `L`

        return relative_positions
