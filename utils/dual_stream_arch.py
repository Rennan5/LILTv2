import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class DualStreamTransformerWithAttention(nn.Module):
    def __init__(self, base_model_name="bert-base-uncased", num_tasks=3, task_heads=None):
        """
        Initialize the Dual-Stream Transformer model with Attention Enhancement.

        Args:
            base_model_name (str): Name of the pre-trained model to use.
            num_tasks (int): Number of tasks for multitask learning.
            task_heads (list of dict): List containing configurations for each task head.
                                       Each dict must specify 'type' ("classification" or "regression")
                                       and 'output_size' (number of output units).
        """
        super(DualStreamTransformerWithAttention, self).__init__()

        # Primary encoder for textual data
        self.text_encoder = AutoModel.from_pretrained(base_model_name)

        # Secondary encoder for structured data
        self.structured_encoder = nn.Sequential(
            nn.Linear(10, 128),  # Example input size (structured data)
            nn.ReLU(),
            nn.Linear(128, self.text_encoder.config.hidden_size)
        )

        # Dual-stream attention enhancement mechanism
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=self.text_encoder.config.hidden_size,
            num_heads=8,
            batch_first=True
        )

        # Fusion layer to combine text and structured streams
        self.fusion_layer = nn.Linear(2 * self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size)

        # Initialize task-specific heads
        self.task_heads = nn.ModuleList()
        for i, task in enumerate(task_heads):
            if task['type'] == 'classification':
                head = nn.Sequential(
                    nn.Linear(self.text_encoder.config.hidden_size, task['output_size']),
                    nn.Softmax(dim=-1)
                )
            elif task['type'] == 'regression':
                head = nn.Linear(self.text_encoder.config.hidden_size, task['output_size'])
            else:
                raise ValueError(f"Unsupported task type: {task['type']}")
            self.task_heads.append(head)

    def forward(self, input_ids, attention_mask, structured_data, task_id):
        """
        Forward pass through the Dual-Stream Transformer model with Attention Enhancement.

        Args:
            input_ids (torch.Tensor): Input token IDs for textual data.
            attention_mask (torch.Tensor): Attention mask for textual data.
            structured_data (torch.Tensor): Input tensor for structured data.
            task_id (int): Index of the task to process.

        Returns:
            torch.Tensor: Output from the selected task head.
        """
        # Textual data encoding
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_hidden_state = text_outputs.last_hidden_state  # All token representations

        # Structured data encoding
        structured_hidden_state = self.structured_encoder(structured_data).unsqueeze(1)  # Add sequence dimension

        # Apply dual-stream attention enhancement
        attn_output, _ = self.attention_layer(
            text_hidden_state, structured_hidden_state, structured_hidden_state
        )

        # Combine the enhanced attention outputs
        combined_state = torch.cat((attn_output[:, 0], structured_hidden_state.squeeze(1)), dim=-1)  # Combine [CLS] and structured
        fused_state = self.fusion_layer(combined_state)

        # Pass through the task-specific head
        output = self.task_heads[task_id](fused_state)
        return output

# Example usage
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Define task configurations
    tasks = [
        {'type': 'classification', 'output_size': 3},  # e.g., sentiment analysis
        {'type': 'classification', 'output_size': 2},  # e.g., binary classification
        {'type': 'regression', 'output_size': 1}       # e.g., score prediction
    ]

    model = DualStreamTransformerWithAttention(base_model_name="bert-base-uncased", num_tasks=3, task_heads=tasks)

    # Sample textual input
    text = "This is a sample input for the model."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Sample structured input
    structured_data = torch.rand((1, 10))  # Example structured data with 10 features

    # Forward pass for task 0
    task_id = 0
    outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], structured_data=structured_data, task_id=task_id)

    print("Output shape:", outputs.shape)
