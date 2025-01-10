import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class LILTv2(nn.Module):
    def __init__(self, base_model_name="bert-base-uncased", num_tasks=3, task_heads=None):
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

        # Initialize task-specific heads
        self.task_heads = nn.ModuleList()
        for i, task in enumerate(task_heads):
            if task['type'] == 'classification':
                head = nn.Sequential(
                    nn.Linear(self.encoder.config.hidden_size, task['output_size']),
                    nn.Softmax(dim=-1)
                )
            elif task['type'] == 'regression':
                head = nn.Linear(self.encoder.config.hidden_size, task['output_size'])
            else:
                raise ValueError(f"Unsupported task type: {task['type']}")
            self.task_heads.append(head)

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

# Example usage
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Define task configurations
    tasks = [
        {'type': 'classification', 'output_size': 3},  # e.g., sentiment analysis
        {'type': 'classification', 'output_size': 2},  # e.g., binary classification
        {'type': 'regression', 'output_size': 1}       # e.g., score prediction
    ]

    model = LILTv2(base_model_name="bert-base-uncased", num_tasks=3, task_heads=tasks)

    # Sample input
    text = "This is a sample input for the model."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    # Forward pass for task 0
    task_id = 0
    outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], task_id=task_id)

    print("Output shape:", outputs.shape)
