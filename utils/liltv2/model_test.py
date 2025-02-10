import torch
import torch.nn as nn
from transformers import AutoModel

class LILTv2(nn.Module):
    def __init__(self, base_model_name="bert-base-uncased", epochs=10, batch_size=32, learning_rate=5e-5, num_tasks=None, task_heads=None, loss_func=nn.CrossEntropyLoss()):
        """
        Initialize the LILTv2 model.

        Args:
            base_model_name (str): Name of the pre-trained model to use.
            num_tasks (int): Number of tasks for multitask learning.
            task_heads (list of dict): List containing configurations for each task head.
                                       Each dict must specify 'type' ("classification" or "regression")
                                       and 'output_size' (number of output units).
        """
        super(LILTv2, self).__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        self.task_heads = task_heads
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.heads = []
        self.loss_func = loss_func
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        if num_tasks is not None:
            self.task_heads = []
            for _ in range(num_tasks):
                self.task_heads.append({'type': 'classification', 'output_size': 2})
        
        # Initialize task-specific heads
        #self.task_heads = nn.ModuleList()
        for task in self.task_heads:
            if type(task) is not dict: break
            if task['type'] == 'classification':
                head = nn.Sequential(
                    nn.Linear(self.encoder.config.hidden_size, task['output_size']),
                    nn.Softmax(dim=-1)
                )
            elif task['type'] == 'regression':
                head = nn.Linear(self.encoder.config.hidden_size, task['output_size'])
            else:
                raise ValueError(f"Unsupported task type: {task['type']}")
            self.heads.append(head)

    def __str__(self):
        res = f'num of tasks: {len(self.task_heads)}\n'
        for task_head in self.task_heads:
            res += f'{task_head['type']}\n'
        return res

    def forward(self, input_ids, attention_mask, task_id):
        """
        Forward pass through the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.
            task_id (int): Index of the task to process.

        Returns:
            torch.Tensor: Output from the selected task head.
        """
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = encoder_outputs.last_hidden_state
        pooled_output = hidden_state[:, 0]  # Use [CLS] token representation

        # Pass through the task-specific head
        output = self.task_heads[task_id](pooled_output)
        return output

    def train_step(self, input_ids, attention_mask, labels, task_id):
        """
        Perform a single training step with backpropagation.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.
            labels (torch.Tensor): Ground truth labels.
            task_id (int): Index of the task to process.

        Returns:
            float: Computed loss value.
        """
        optimizer = self.optimizer
        loss_fn = self.loss_func
        
        optimizer.zero_grad()
        outputs = self.forward(input_ids, attention_mask, task_id)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        return loss.item()

    def dual_stream_pretrain(self, text_inputs, visual_inputs, attention_mask):
        """
        Dual-Stream Transformer Architecture for dataset pretraining.

        Args:
            text_inputs (torch.Tensor): Input token IDs for text.
            visual_inputs (torch.Tensor): Input tensor for visual data.
            attention_mask (torch.Tensor): Attention mask for text inputs.

        Returns:
            float: Computed loss value for the pretraining step.
        """
        num_epochs = self.epochs
        batch_size = self.batch_size
        learning_rate = self.learning_rate
        optimizer = self.optimizer
        loss_fn = self.loss_func
        
        optimizer.param_groups[0]['lr'] = learning_rate
        dataset_size = text_inputs.shape[0]
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for i in range(0, dataset_size, batch_size):
                batch_text = text_inputs[i:i+batch_size]
                batch_visual = visual_inputs[i:i+batch_size]
                batch_attention = attention_mask[i:i+batch_size]
                
                optimizer.zero_grad()
                
                # Encode text
                text_outputs = self.encoder(input_ids=batch_text, attention_mask=batch_attention)
                text_hidden = text_outputs.last_hidden_state[:, 0]
                
                # Encode visual data (Placeholder: Replace with actual visual encoder)
                visual_hidden = torch.nn.Linear(batch_visual.shape[-1], self.encoder.config.hidden_size)(batch_visual)
                
                # Fusion of both modalities
                fused_representation = torch.cat((text_hidden, visual_hidden), dim=-1)
                fused_representation = torch.nn.Linear(fused_representation.shape[-1], self.encoder.config.hidden_size)(fused_representation)
                
                # Compute loss
                loss = loss_fn(fused_representation, text_hidden)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/dataset_size}")
        
        return epoch_loss / dataset_size

    def train_model(self, train_loader, task_id):
        """
        Train the model for multiple epochs.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            task_id (int): Index of the task to process.

        """
        num_epochs = self.epochs
        optimizer = self.optimizer
        loss_fn = self.loss_func
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch in train_loader:
                input_ids, attention_mask, labels = batch
                loss = self.train_step(input_ids, attention_mask, labels, task_id, optimizer, loss_fn)
                total_loss += loss
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss}")
