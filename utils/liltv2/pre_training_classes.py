import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import Trainer,get_scheduler
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

class MVLMTask(nn.Module):
    def __init__(self, model, tokenizer, vocab_size, mask_prob=0.15, lr=5e-5):
        """
        Classe para treinar a Masked Visual-Language Modeling (MVLM).

        Args:
            model: O modelo LiLTv2 ou outro modelo baseado em Transformer.
            tokenizer: Tokenizer do LayoutLMv3 ou equivalente.
            vocab_size: Tamanho do vocabulário.
            mask_prob: Probabilidade de mascaramento (padrão 15%).
            lr: Taxa de aprendizado para o otimizador.
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.mask_prob = mask_prob
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)

    def mask_tokens(self, input_ids):
        """
        Aplica a máscara nos tokens de entrada, seguindo a regra 80%-10%-10%.
        """
        device = input_ids.device
        labels = input_ids.clone()

        probability_matrix = torch.full(input_ids.shape, self.mask_prob, device=device)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        mask_token_id = self.tokenizer.convert_tokens_to_ids("[MASK]")
        
        masked_input = input_ids.clone()
        masked_input[masked_indices] = mask_token_id

        # 10% dos tokens mascarados -> substituídos por palavras aleatórias
        random_tokens = torch.randint(self.vocab_size, input_ids.shape, dtype=torch.long, device=device)
        random_indices = torch.bernoulli(torch.full(input_ids.shape, 0.10, device=device)).bool() & masked_indices
        masked_input[random_indices] = random_tokens[random_indices]

        labels[~masked_indices] = -100  # Ignorar os tokens não mascarados
        return masked_input, labels

    def forward(self, input_ids, attention_mask):
        """
        Executa um forward pass com a MVLM.
        """
        masked_input, labels = self.mask_tokens(input_ids)
        outputs = self.model(masked_input, attention_mask=attention_mask)
        logits = outputs.logits  # Saída do modelo
        loss = nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, self.vocab_size), labels.view(-1))
        return loss

    def train_task(self, dataloader, num_epochs, output_dir):
        """
        Treina a MVLM e salva o estado do modelo.
        """
        self.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in dataloader:
                input_ids, attention_mask = batch["input_ids"].to(self.model.device), batch["attention_mask"].to(self.model.device)
                
                self.optimizer.zero_grad()
                loss = self(input_ids, attention_mask)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        torch.save(self.state_dict(), f"{output_dir}/mvlm_task.pth")
        print("Treinamento concluído e modelo salvo!")

class WPATask(nn.Module):
    def __init__(self, model, patch_size, img_size, lr=5e-5):
        """
        Implementação da Word-Patch Alignment (WPA).

        Args:
            model: O modelo LiLTv2 ou outro modelo baseado em Transformer.
            patch_size: Tamanho do patch de imagem.
            img_size: Tamanho da imagem de entrada.
            lr: Taxa de aprendizado para o otimizador.
        """
        super().__init__()
        self.model = model
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2
        self.classifier = nn.Linear(model.config.hidden_size, 2)  # Binário: mascarado ou não
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)

    def mask_patches(self, patch_embeddings, mask_prob=0.15):
        """
        Aplica a máscara nos patches de imagem.

        Args:
            patch_embeddings (torch.Tensor): Embeddings dos patches da imagem.
            mask_prob (float): Probabilidade de mascaramento (15% por padrão).

        Returns:
            masked_patches (torch.Tensor): Patches mascarados.
            labels (torch.Tensor): Rótulos indicando quais patches foram mascarados.
        """
        batch_size, num_patches, hidden_dim = patch_embeddings.shape
        labels = torch.zeros(batch_size, num_patches, dtype=torch.long, device=patch_embeddings.device)
        
        # Seleciona patches a serem mascarados
        mask = torch.rand(batch_size, num_patches, device=patch_embeddings.device) < mask_prob
        labels[mask] = 1  # Define os patches mascarados como 1

        masked_patches = patch_embeddings.clone()

        # 80% dos patches mascarados -> substituídos por um token especial de máscara
        mask_token = torch.zeros_like(masked_patches)  # Patches zerados atuam como token de máscara
        masked_patches[mask] = mask_token[mask]

        # 10% dos patches mascarados -> substituídos por patches aleatórios
        random_patches = torch.randn_like(patch_embeddings)
        random_indices = torch.rand(batch_size, num_patches, device=patch_embeddings.device) < 0.10
        random_indices &= mask  # Apenas entre os mascarados
        masked_patches[random_indices] = random_patches[random_indices]

        # 10% restantes permanecem inalterados (já feito pela inicialização)
        return masked_patches, labels

    def forward(self, patch_embeddings):
        """
        Forward pass da tarefa WPA.

        Args:
            patch_embeddings (torch.Tensor): Embeddings dos patches de imagem.

        Returns:
            loss (torch.Tensor): Perda da tarefa WPA.
        """
        masked_patches, labels = self.mask_patches(patch_embeddings)
        logits = self.classifier(masked_patches)
        loss = nn.CrossEntropyLoss()(logits.view(-1, 2), labels.view(-1))
        return loss

    def train_task(self, dataloader, num_epochs, output_dir):
        """
        Treina a tarefa WPA e salva o estado do modelo.

        Args:
            dataloader: DataLoader contendo os embeddings dos patches de imagem.
            num_epochs: Número de épocas para treinamento.
            output_dir: Diretório onde o modelo será salvo.
        """
        self.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in dataloader:
                patch_embeddings = batch["patch_embeddings"].to(self.model.device)

                self.optimizer.zero_grad()
                loss = self(patch_embeddings)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        torch.save(self.state_dict(), f"{output_dir}/wpa_task.pth")
        print("Treinamento concluído e modelo salvo!")

class KPLTask(nn.Module):
    def __init__(self, model, grid_size, num_classes, lr=5e-5):
        """
        Implementação da Key Point Location (KPL).

        Args:
            model: O modelo base (LiLTv2 ou similar).
            grid_size: Tamanho da grade para quantização das posições.
            num_classes: Número de regiões na grade (grid_size * grid_size).
            lr: Taxa de aprendizado.
        """
        super().__init__()
        self.model = model
        self.grid_size = grid_size
        self.num_classes = num_classes  # Quantidade de regiões para classificar os key points
        self.classifier = nn.Linear(model.config.hidden_size, num_classes * 3)  # 3 key points por bounding box
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)

    def mask_bounding_boxes(self, bounding_boxes, mask_prob=0.15):
        """
        Aplica máscara nos bounding boxes com estratégia similar ao MVLM.

        Args:
            bounding_boxes (torch.Tensor): Tensor de bounding boxes (B, T, 6).
            mask_prob (float): Probabilidade de mascaramento.

        Returns:
            masked_boxes (torch.Tensor): Bounding boxes mascarados.
            labels (torch.Tensor): Rótulos indicando as regiões dos key points.
        """
        batch_size, num_boxes, _ = bounding_boxes.shape
        labels = bounding_boxes.clone()
        device = bounding_boxes.device

        print(f"bounding_boxes.shape antes de clonar: {bounding_boxes.shape}")  

        # 🔹 Certifica que bounding_boxes tem shape correto (B, T, 6)
        if bounding_boxes.shape[-1] > 6:
            bounding_boxes = bounding_boxes[:, :, :6]

        print(f"bounding_boxes.shape corrigido: {bounding_boxes.shape}")  

        # Seleciona quais bounding boxes serão mascarados
        mask = torch.rand(batch_size, num_boxes, device=device) < mask_prob
        masked_boxes = bounding_boxes.clone()

        # 80% → Substituídos por um token especial de máscara
        mask_token = torch.zeros_like(masked_boxes)
        masked_boxes[mask] = mask_token[mask]

        # 10% → Substituídos por bounding boxes aleatórios do batch
        random_indices = torch.randint(0, batch_size, (batch_size,), device=device).unsqueeze(-1).unsqueeze(-1)
        random_boxes = bounding_boxes[random_indices, :, :]

        # Corrigindo random_indices para corresponder ao formato correto (B, T, 6)
        random_indices = torch.bernoulli(torch.full((batch_size, num_boxes, 1), 0.10, device=device)).bool().expand(-1, -1, 6) & mask.unsqueeze(-1)

        # ✅ Usa torch.where para evitar erro de shape
        masked_boxes = torch.where(random_indices, random_boxes, masked_boxes)

        return masked_boxes, labels


    def quantize_to_grid(self, keypoints, img_size):
        """
        Quantiza os key points para a grade de regiões.

        Args:
            keypoints (torch.Tensor): Tensor de coordenadas dos key points (B, T, 3, 2).
            img_size (int): Dimensão da imagem.

        Returns:
            torch.Tensor: Índices das regiões na grade.
        """
        grid_step = max(1, img_size // self.grid_size)  # 🔹 Evita divisão por zero
        indices = (keypoints // grid_step).long()  # Mapeia as coordenadas para células da grade

        # 🔹 Garante que os índices estejam dentro do intervalo válido
        indices[..., 0] = indices[..., 0].clamp(0, self.grid_size - 1)
        indices[..., 1] = indices[..., 1].clamp(0, self.grid_size - 1)

        return indices[..., 0] * self.grid_size + indices[..., 1] 

    def forward(self, bounding_boxes, img_size, attention_mask, input_ids):
        _, labels = self.mask_bounding_boxes(bounding_boxes)

        device = self.model.device  
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

        with torch.no_grad():
            outputs = self.model.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        self.classifier.to(device)  # Move `self.classifier` para `device`
        logits = self.classifier(outputs.to(device))  # (B, T, num_classes * 3)

        # Reformatar os labels dos key points
        keypoints = labels[:, :, [0, 1, 2, 3, 4, 5]].reshape(labels.shape[0], labels.shape[1], 3, 2)
        target_labels = self.quantize_to_grid(keypoints, img_size)

        # 🔹 Verifica valores inválidos antes de calcular a loss
        if (target_labels < 0).any() or (target_labels >= self.num_classes).any():
            print(f"[ERRO] target_labels contém valores fora do intervalo esperado: {target_labels}")

        target_labels = target_labels.clamp(0, self.num_classes - 1)  # 🔹 Garante que os valores estejam no intervalo correto

        loss = nn.CrossEntropyLoss()(logits.view(-1, self.num_classes), target_labels.view(-1))

        return logits, loss

    def train_task(self, dataloader, num_epochs, output_dir, img_size):
        """
        Treina a tarefa KPL e salva o estado do modelo.

        Args:
            dataloader: DataLoader contendo os bounding boxes e dados de entrada.
            num_epochs: Número de épocas para treinamento.
            output_dir: Diretório onde o modelo será salvo.
            img_size: Dimensão da imagem usada para quantização.
        """
        self.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in dataloader:
                bounding_boxes = batch["bounding_boxes"].to(self.model.device)

                self.optimizer.zero_grad()
                loss = self.forward(bounding_boxes, img_size)  # Chama a função forward explicitamente
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        torch.save(self.state_dict(), f"{output_dir}/kpl_task.pth")
        print("Treinamento concluído e modelo salvo!")

class RPCTask(nn.Module): 
    def __init__(self, model, input_dim, hidden_dim, num_directions=8, num_distances=5, lr=1e-3):
        """
        Implementação da Relative Position Classification (RPC).

        Args:
            model: O modelo base (LiLTv2 ou similar).
            input_dim: Dimensão dos embeddings das caixas de texto.
            hidden_dim: Dimensão intermediária da FFN.
            num_directions: Número de direções para classificação (8).
            num_distances: Número de classes para distância (ajustável).
            lr: Taxa de aprendizado.
        """
        super().__init__()
        self.model = model
        self.ffn = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.direction_classifier = nn.Linear(hidden_dim, num_directions)
        self.distance_classifier = nn.Linear(hidden_dim, num_distances)
        self.optimizer = optim.AdamW(self.parameters(), lr=lr)

    def forward(self, box1, box2):
        """
        Forward pass da tarefa RPC.

        Args:
            box1 (torch.Tensor): Representação do primeiro bounding box (B, input_dim).
            box2 (torch.Tensor): Representação do segundo bounding box (B, input_dim).

        Returns:
            direction_logits (torch.Tensor): Logits para a classificação de direção (B, num_directions).
            distance_logits (torch.Tensor): Logits para a classificação de distância (B, num_distances).
        """
        x = torch.cat((box1, box2), dim=-1)  # Concatenação das representações
        x = self.ffn(x)  # Passa pelo FFN
        direction_logits = self.direction_classifier(x)
        distance_logits = self.distance_classifier(x)
        return direction_logits, distance_logits

    def train_task(self, dataloader, num_epochs, output_dir):
        """
        Treina a RPC e salva o estado do modelo.

        Args:
            dataloader: DataLoader contendo os pares de bounding boxes e seus rótulos.
            num_epochs: Número de épocas para treinamento.
            output_dir: Diretório onde o modelo será salvo.
        """
        self.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in dataloader:
                box1, box2 = batch["box1"].to(self.model.device), batch["box2"].to(self.model.device)
                direction, distance = batch["direction"].to(self.model.device), batch["distance"].to(self.model.device)

                direction = direction.long()
                distance = distance.long()

                if direction.min() < 0 or direction.max() >= self.direction_classifier.out_features:
                    raise ValueError(f"Invalid direction labels: {direction}")
                if distance.min() < 0 or distance.max() >= self.distance_classifier.out_features:
                    raise ValueError(f"Invalid distance labels: {distance}")

                self.optimizer.zero_grad()
                direction_logits, distance_logits = self.forward(box1, box2)

                loss = nn.CrossEntropyLoss()(direction_logits, direction) + nn.CrossEntropyLoss()(distance_logits, distance)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        torch.save(self.state_dict(), f"{output_dir}/rpc_task.pth")
        print("Treinamento concluído e modelo salvo!")