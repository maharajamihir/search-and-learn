import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from peft import LoraConfig, get_peft_model
import argparse
import msgpack
from pathlib import Path
import wandb
from torch.utils.data import random_split





class DifficultyHeadMLP(nn.Module):
    def __init__(self, lm_name='meta-llama/Llama-3.2-1B-Instruct', mlp_hidden_size=128):
        super(DifficultyHeadMLP, self).__init__()
        # Load pre-trained LLM
        self.tokenizer = AutoTokenizer.from_pretrained(lm_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lm = AutoModelForCausalLM.from_pretrained(lm_name)
        self.method = "mlp"
        self.cache = {}

        # Freeze the language model parameters
        for param in self.lm.parameters():
            param.requires_grad = False

        # MLP for difficulty prediction
        self.mlp = nn.Sequential(
            nn.Linear(self.lm.config.hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask=None):
        # Get hidden states from the LM
        with torch.no_grad():
            outputs = self.lm(input_ids, attention_mask=attention_mask, output_hidden_states=True)

        last_indices = input_ids.ne(self.tokenizer.pad_token_id).sum(dim=1) - 1
        last_token_hidden_state = outputs.hidden_states[-1][torch.arange(input_ids.size(0)), last_indices, :]
        # last_token_hidden_state = outputs.hidden_states[-1][:, -1, :]  # (batch_size, hidden_dim)
        difficulty = self.mlp(last_token_hidden_state)  # (batch_size, 1)

        return difficulty

class DifficultyHeadLoRA(nn.Module):
    def __init__(self, lm_name='meta-llama/Llama-3.2-1B-Instruct', lora_r=4):
        super(DifficultyHeadLoRA, self).__init__()

        # Load pre-trained LLM
        self.tokenizer = AutoTokenizer.from_pretrained(lm_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lm = AutoModelForCausalLM.from_pretrained(lm_name)
        self.method = "lora"

        # Configure LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]        
            )
        self.lm = get_peft_model(self.lm, lora_config)

        # Additional linear layer for difficulty prediction
        self.fc = nn.Linear(self.lm.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None, inference=False):
        # Handle training and inference separately
        if inference:
            with torch.no_grad():
                outputs = self.lm(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        else:
            outputs = self.lm(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        last_token_hidden_state = outputs.hidden_states[-1][:, -1, :]  # (batch_size, hidden_dim)
        difficulty = self.sigmoid(self.fc(last_token_hidden_state))  # (batch_size, 1)

        return difficulty  # Only return difficulty



# Define a simple dataset
class SimpleDataset(Dataset):
    def __init__(self, tokenizer, texts, difficulty, max_length=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.difficulty = torch.tensor(difficulty, dtype=torch.float32)
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        input_ids = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length
        ).input_ids.squeeze(0)
        difficulty = self.difficulty[idx]
        return input_ids, difficulty


def train(model, train_dataloader, val_dataloader, epochs=100, lr=1e-4):
    wandb.init(project="difficulty-head-training", config={"epochs": epochs, "learning_rate": lr, "method": model.method})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # Assuming a regression task for difficulty

    for epoch in range(epochs):
        # Training loop
        model.train()
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            input_ids, difficulty_target = batch
            input_ids = input_ids.to(device)
            difficulty_target = difficulty_target.to(device)
            difficulty_pred = model(input_ids)
            difficulty_pred = difficulty_pred.view_as(difficulty_target)
            loss = criterion(difficulty_pred, difficulty_target)
            loss.backward()
            optimizer.step()
            # Log the training loss to wandb per step
            wandb.log({"epoch": epoch + 1, "step": step + 1, "train_loss": loss.item()})
            print(f"Epoch {epoch+1}, Step {step+1}, Train Loss: {loss.item()}")

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_dataloader:
                input_ids, difficulty_target = val_batch
                input_ids = input_ids.to(device)
                difficulty_target = difficulty_target.to(device)
                difficulty_pred = model(input_ids)
                difficulty_pred = difficulty_pred.view_as(difficulty_target)
                val_loss += criterion(difficulty_pred, difficulty_target).item()

        val_loss /= len(val_dataloader)
        wandb.log({"epoch": epoch + 1, "val_loss": val_loss})
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss}")

        # Save the model checkpoint
        checkpoint_path = Path(__file__).parent / f"{model.method}_checkpoint.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")


if __name__ == "__main__":

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Difficulty Head Model")
    parser.add_argument("--method", type=str, choices=["LoRA", "MLP"], required=True, help="Method to use: LoRA or MLP")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    args = parser.parse_args()

    # Use the parsed arguments
    method = args.method
    file_path = args.data_path

    # Initialize the model
    if method == "LoRA":
        model = DifficultyHeadLoRA()
    elif method == "MLP":
        model = DifficultyHeadMLP()
    else:
        raise ValueError(f"Unsupported method: {method}")


    print(f"Loading {file_path}, this might take a while...")
    file_size_in_bytes = Path(file_path).stat().st_size
    if file_size_in_bytes >= 1e9:
        file_size = file_size_in_bytes / 1e9
        size_unit = "GB"
    else:
        file_size = file_size_in_bytes / 1e3
        size_unit = "KB"
    print(f"The size of the file is: {file_size:.2f} {size_unit}")
    with open(file_path, 'rb') as f:
        results = msgpack.unpackb(f.read())

    texts = [r["problem"] for r in results]
    difficulty = [r["pass@1"] for r in results]

    # Create a dataset and split into train and validation
    dataset = SimpleDataset(model.tokenizer, texts, difficulty)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    train(model, train_dataloader, val_dataloader)

    # Save the model to a checkpoint
    checkpoint_path = "difficulty_head_checkpoint.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")
