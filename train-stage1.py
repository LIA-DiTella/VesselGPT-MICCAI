import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils.rnn import pad_sequence

import os
import numpy as np

from Etapa1.modelsMultitalk.stage1Emma import VQAutoEncoder
from Etapa1.metrics.loss import calc_vq_loss
from Etapa1.base.utilities import AverageMeter
import wandb
from funciones import IntraDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

torch.manual_seed(12)
np.random.seed(12)


class Args:
    def __init__(self):
        
        self.quant_loss_weight = 1.
        self.base_lr = 0.0001
        self.StepLR = True
        #self.warmup_steps = 1
        self.step_size = 200
        self.gamma = 0.9
        #self.weight_decay = 0.002
       
       
    
# Instantiate the arguments
args = Args()


def collate_fn_intra(batch):
    # Sort sequences by length in descending order
    batch = sorted(batch, key=lambda x: x.shape[0], reverse=True)
    
    # Extract lengths
    lengths = [x.shape[0] for x in batch]
    
    # Pad sequences
    padded_trees = pad_sequence(batch, batch_first=True, padding_value=0.0)

    # Create mask: True for valid data, False for padding
    tree_mask = torch.zeros(padded_trees.shape[:2], dtype=torch.bool)
    for i, l in enumerate(lengths):
        tree_mask[i, :l] = 1

    return padded_trees, tree_mask

def load_file_list(txt_path):
    with open(txt_path, 'r') as f:
        return [line.strip() for line in f]

# Load paths
train_files = load_file_list("splits/train_files.txt")
val_files = load_file_list("splits/val_files.txt")
test_files = load_file_list("splits/test_files.txt")

root_dir = "Datos/Aneux+Intra-splines/zero-root"

intra_dataset = IntraDataset(train_files, mode="post_order", p=15, root_dir=root_dir) #mode es el modo al que quiero que transforme
#intra_dataset = IntraDataset(INTRA_FOLDER, mode="pre_order")
data_loader = DataLoader(intra_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn_intra)

#dataset de validacion
#VAL_FOLDER = "Datos/AneuxSplines/zero-root/p15/val/"
val_dataset = IntraDataset(val_files, mode = "post_order", p=15, root_dir=root_dir) #mode es el modo al que quiero que transforme
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn_intra)

def validate(val_loader, model, loss_fn, epoch, cfg):
    accumulated_loss = 0
    accumulated_rec = 0
    accumulated_quant = 0
    model.eval()

    with torch.no_grad():
        for batch in val_loader:
                inputs = tuple(x.to(device) for x in batch)
                padded_trees, tree_mask = inputs
            
                out, vq_loss = model(padded_trees, mask=tree_mask)
                vq_loss = vq_loss.mean()
                              
                # LOSS
                loss, loss_details = loss_fn(out, padded_trees, quant_loss=vq_loss, quant_loss_weight=cfg.quant_loss_weight)
                accumulated_loss += loss
                accumulated_rec += loss_details[0]
                accumulated_quant += loss_details[1]
        

        # Calculate the average loss over the entire dataset
        avg_loss = accumulated_loss / len(val_loader)
        rec_loss = accumulated_rec / len(val_loader)
        quant_loss = accumulated_quant / len(val_loader)
   

    return avg_loss, rec_loss, quant_loss

wandb.login(anonymous="must")
wandb.login(key="2511bccb1c20c8149e91d2ff7ad5b57fab7df870")

wandb.init(project="vqvae-mesh")

def save_best_model(model, optimizer, epoch, loss, best_loss, model_save_path="best_model.pth"):
    """
    Save the model if the current loss is better than the best recorded loss.
    
    Args:
        model (torch.nn.Module): The model being trained.
        optimizer (torch.optim.Optimizer): The optimizer used in training.
        epoch (int): The current epoch number.
        loss (float): The current epoch's loss.
        best_loss (float): The best loss recorded so far.
        model_save_path (str): Path to save the best model.
    
    Returns:
        float: The updated best loss (could be the same or updated if the model improved).
    """
    if loss < best_loss:
        #print(f"Epoch [{epoch+1}], New best model found! Loss: {loss:.4f}")
        best_loss = loss
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, model_save_path)
    return best_loss


# Sample training loop
num_epochs = 5000  # Define number of epochs
best_loss = float('inf')  # Initialize to a very high value

model = VQAutoEncoder(5)

model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr)
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: max(1e-4, min(1.0, epoch / 100)))
scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of parameters: {total_params}")

rec_loss_list = []
quant_loss_list = []
pp_list = []
total_loss_list = []

wandb.watch(model, log="all")

initial_quant_loss_weight = 100.0  # Starting weight for quantization loss
final_quant_loss_weight = 2.     # Final weight for quantization loss        
decay_epochs = 300

for epoch in range(num_epochs):
    
    model.train()
    rec_loss_meter = AverageMeter()
    quant_loss_meter = AverageMeter()
    total_loss_meter = AverageMeter()
    pp_meter = AverageMeter()
    accum = 0
    for batch in data_loader:
        inputs = tuple(x.to(device) for x in batch)
        padded_trees, tree_mask = inputs
       
        outputs, vq_loss = model(padded_trees, mask=tree_mask)
        vq_loss = vq_loss.mean()
          
        # Calculate loss
        loss, loss_details = calc_vq_loss(outputs, padded_trees, quant_loss=vq_loss, quant_loss_weight=args.quant_loss_weight)

            
        #############
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #accum += loss

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            
        for m, x in zip([rec_loss_meter, quant_loss_meter, total_loss_meter],
                    [loss_details[0], loss_details[1], loss]):
            m.update(x.item(), 1)

        

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    avg_loss_val, rec_loss_val, quant_loss_val = validate(val_loader, model, calc_vq_loss, epoch, args)
    # Append averaged losses after each epoch
    rec_loss = rec_loss_meter.avg
    quant_loss = quant_loss_meter.avg
    total_loss = total_loss_meter.avg #aca tengo la total loss promedio de la epoch
    
    wandb.log({
        "Epoch": epoch + 1,
        "Reconstruction Loss": rec_loss,
        "Quantization Loss": quant_loss,
        "Total Loss": total_loss,
        "Total loss val": avg_loss_val,
        "Reconstruction loss val": rec_loss_val,
        "Quantization loss val": quant_loss_val,
        "Learning Rate": optimizer.param_groups[0]['lr']
    })
    # Append to respective lists
    rec_loss_list.append(rec_loss)
    quant_loss_list.append(quant_loss)

    
    # Call the function to check if we need to save the model
    #best_loss = save_best_model(model, optimizer, epoch, rec_loss_val, best_loss, "modelos-entrenados/aneurisk-limpio-splines-15.pth")
    best_loss = save_best_model(model, optimizer, epoch, rec_loss_val, best_loss, "models/stage1/aneux+intra/best-model-nuevoquant-zeroroot.pth")

    # Print learning rate and losses every 10 epochs
    if epoch % 1 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {total_loss:.4f}, Rec Loss: {rec_loss:.5f}, Quant Loss: {quant_loss:.5f}, Perplexity: {pp_meter.avg:.4f}, Learning Rate: {current_lr:.6f}")
        #print(f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {loss:.4f}, Learning Rate: {current_lr:.8f}")

    # Step the scheduler
    scheduler.step()
    if epoch == 280:
        # Reinitialize with a larger step size
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=1)