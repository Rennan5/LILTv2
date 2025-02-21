import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_scheduler
from tqdm import tqdm

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
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, token_embeds, patch_embeds):
        T, P = token_embeds.size(1), patch_embeds.size(1)
        token_embeds = token_embeds.unsqueeze(2).expand(-1, -1, P, -1)
        patch_embeds = patch_embeds.unsqueeze(1).expand(-1, T, -1, -1)
        pairwise_features = torch.cat([token_embeds, patch_embeds], dim=-1)
        alignment_scores = self.alignment_score(pairwise_features).squeeze(-1)
        return alignment_scores

    def train_task(self, train_dataset, val_dataset, training_args):
        """
        Treina a tarefa WPA (Word-Patch Alignment).
        """
        print("Treinando WPA...")  

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        # Otimizador e função de perda
        optimizer = optim.AdamW(self.parameters(), lr=training_args.learning_rate)
        loss_fn = nn.BCEWithLogitsLoss()

        num_training_steps = training_args.num_train_epochs * len(train_dataset)
        lr_scheduler = get_scheduler(
            "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )

        for epoch in range(training_args.num_train_epochs):
            self.train()
            total_loss = 0

            for batch in tqdm(train_dataset, desc=f"Época {epoch + 1}/{training_args.num_train_epochs} - Treinamento WPA"):
                token_embeds = batch["token_embeds"].to(device)  # (B, T, embed_dim)
                patch_embeds = batch["patch_embeds"].to(device)  # (B, P, embed_dim)
                alignment_labels = batch["alignment_labels"].to(device)  # (B, T, P)

                optimizer.zero_grad()
                alignment_preds = self.forward(token_embeds, patch_embeds)
                loss = loss_fn(alignment_preds, alignment_labels)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                total_loss += loss.item()
                
        print("Treinamento de WPA finalizado!")
    
class KPLTask(nn.Module):
    def __init__(self, embed_dim):
        """
        Implementa a tarefa KPL (Key Point Location).
        """
        super(KPLTask, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 4)  # Previsão de (x1, y1, x2, y2)
        )

    def forward(self, text_embeds):
        return self.mlp(text_embeds)

    def train_task(self, train_dataset, val_dataset, training_args):
        """
        Treina a tarefa KPL (Key Point Location).
        """
        print("Treinando KPL...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        # Otimizador e função de perda
        optimizer = optim.AdamW(self.parameters(), lr=training_args.learning_rate)
        loss_fn = nn.MSELoss()  # Para prever coordenadas contínuas

        num_training_steps = training_args.num_train_epochs * len(train_dataset)
        lr_scheduler = get_scheduler(
            "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )

        for epoch in range(training_args.num_train_epochs):
            self.train()
            total_loss = 0

            for batch in tqdm(train_dataset, desc=f"Época {epoch + 1}/{training_args.num_train_epochs} - Treinamento KPL"):
                text_embeds = batch["text_embeds"].to(device)  # (B, T, embed_dim)
                bbox_labels = batch["bbox_labels"].to(device)  # (B, T, 4)

                optimizer.zero_grad()
                bbox_preds = self.forward(text_embeds)
                loss = loss_fn(bbox_preds, bbox_labels)
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                total_loss += loss.item()

        print("Treinamento de KPL finalizado!")


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