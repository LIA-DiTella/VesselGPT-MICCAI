import torch
from transformers import GPT2Config, GPT2LMHeadModel
from torch.utils.data import DataLoader, Dataset
from transformers import get_scheduler
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
from funciones import *
from torch.nn.utils.rnn import pad_sequence

import warnings
warnings.filterwarnings("ignore")

import wandb
import gc
TRAIN = True
WANDB_UPLOAD = True

vocab_size = 258        # 256 : EOS token , 257 : pad token
max_size = 2352 + 2
pad_token = 257
eos_token = 256

epochs = 50000
lr = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
class TokenDataset(Dataset):

    def __init__(self, folder_path, n_samples):

        self.samples = []
        self._load_files(folder_path, n_samples)

    def _load_files(self, folder_path, n_samples):

        files = os.listdir(folder_path)#[:n_samples]

        for file_name in files:

            file_path = os.path.join(folder_path, file_name)
            self.samples.append(torch.load(file_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        eos = torch.tensor([eos_token])
        seq = torch.cat((eos, self.samples[idx], eos))

        return torch.tensor(seq, dtype = torch.long)

def custom_collate(batch, pad_token_id = 257):
    return pad_sequence(batch, batch_first = True, padding_value = pad_token_id)

def create_attention_mask(batch, pad_token_id):
    return (batch != pad_token_id).long()  # 1 for real tokens, 0 for padding

def create_gpt2_model(vocab_size, max_size, pad_token):
    
    config = GPT2Config(

        vocab_size = vocab_size,
        n_embd = 512,  # Size of embeddings
        n_layer = 6,   # Number of layers
        n_head = 8,    # Number of attention heads
        n_positions = max_size,  # Increase max sequence length
        n_ctx = max_size, 
        pad_token_id = pad_token
    )

    return GPT2LMHeadModel(config)

dataset_name = "./datasets/dataset_aneurisk_zero_rot/tokenized"

dataset = TokenDataset(dataset_name, n_samples = 4376)
dataloader = DataLoader(dataset, batch_size = 1, collate_fn = custom_collate, shuffle = False)

print(len(dataset))
max = 0
for seq in dataset: 
    if len(seq) > max: max = len(seq)

print("Largest sequence length :", max)

avg_losses = []
errors = []

if TRAIN:

    if WANDB_UPLOAD:

        wandb.login(key = "451637d95c22df4568c6f5a268e37071bc14547b")
        wandb.init(project = "gpt2", entity = "vesselgpt")

        wandb.config.update({
            "learning_rate": lr,
            "epochs": epochs,
            "dataset": dataset_name,
            "dataset_size": len(dataset),
            "vocab_size": vocab_size
        })

    model = create_gpt2_model(vocab_size, max_size, pad_token)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(dataloader) * epochs)
    best_loss = float('inf') 

    model.train()

    for epoch in range(epochs): 

        total_loss = 0
        for batch in dataloader:

            batch = batch.to(device)
            attention_mask = create_attention_mask(batch, pad_token).to(device)  

            outputs = model(batch, labels = batch, attention_mask = attention_mask)

            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            del outputs, loss, batch
            gc.collect()

        avg_loss = total_loss / len(dataloader)
        avg_losses.append(avg_loss)

        print(f"Epoch {epoch} | Avg Loss: {avg_loss}")
        if WANDB_UPLOAD: wandb.log({"epoch": epoch, "avg_loss": avg_loss})

        # save best model

        current_lr = optimizer.param_groups[0]['lr']
        best_loss = save_best_model_gpt2(model, optimizer, epoch, avg_loss, best_loss, "models/gpt2/gpt2-new")