import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from pathlib import Path


import os
import numpy as np
import Etapa1.modelsMultitalk.stage1_vocaset as models
from Etapa1.modelsMultitalk.stage1_vocaset import VQAutoEncoder
from Etapa1.metrics.loss import calc_vq_loss
from Etapa1.base.utilities import AverageMeter

class Args:
    def __init__(self):
        # LOSS settings
        self.quant_loss_weight = 1.

        # NETWORK settings
        #self.arch = 'stage1_vocaset'
        self.in_dim = 4
        self.hidden_size = 1024
        self.num_hidden_layers = 6
        self.num_attention_heads = 8
        self.intermediate_size = 1536
        self.window_size = 1
        self.quant_factor = 0
        self.face_quan_num = 16
        self.neg = 0.2
        self.INaffine = False

        # VQuantizer settings
        self.n_embed = 256
        self.zquant_dim = 64#64

        # TRAIN settings
        self.batch_size = 1  # batch size for training
        self.batch_size_val = 1  # batch size for validation during training
        self.base_lr = 0.0001
        self.StepLR = True
        self.poly_lr = False
        self.epochs = 50000
        self.step_size = 200
        self.gamma = 0.9


        ##stage 2
        self.device = 'cuda'  # or 'cpu'
        self.feature_dim = 128  # dimension for the feature after audio encoding
        self.vertice_dim = 31  # number of vertices * 3 (e.g., V * 3 for 3D coordinates)
        self.n_head = 8  # number of attention heads in the transformer decoder
        self.num_layers = 6  # number of layers in the transformer decoder
        self.period = 2#100  # period for positional encoding
        self.vqvae_pretrained_path = 'modelos-entrenados/major-oath.pth'  # path to pretrained VQ-VAE

class Tree:

    def __init__(self, data, right = None, left = None):

        self.id = id(self)
        self.data = data

        self.right = right
        self.left = left

def deserialize_post_order(serial):

    serial = serial.copy()

    def post_order(serial):

        if serial[-4:] == [0.0] * 4:
            for i in range(4): serial.pop()
            return None
        
        data = {}

        data["r"] = serial.pop()
        data["z"] = serial.pop()
        data["y"] = serial.pop()
        data["x"] = serial.pop()

        tree = Tree(data)

        tree.right = post_order(serial)
        tree.left = post_order(serial)
        
        return tree    
    
    return post_order(serial)

def deserialize_pre_order(serial):
    
    serial = serial.copy()

    if len(serial) > 0:

        if serial[:4] != [0.0] * 4:
            
            data = {}

            data["x"] = serial.pop(0)
            data["y"] = serial.pop(0)
            data["z"] = serial.pop(0)
            data["r"] = serial.pop(0)

            tree = Tree(data)

            left, ret = deserialize_pre_order(serial)
            right, ret = deserialize_pre_order(ret)

            tree.left = left
            tree.right = right

            return tree, ret

        else:
            return None, serial[4:]
        
    else:
        return None, []

def serialize_pre_order(tree, k):

    if tree == None: return [0.0] * k
    return list(tree.data.values())[::-1] + serialize_pre_order(tree.left) + serialize_pre_order(tree.right)

def deserialize(serial, mode = "pre_order"):

    if mode == "pre_order": return deserialize_pre_order(serial)[0]
    if mode == "post_order": return deserialize_post_order(serial)

    print("UNSUPPORTED DESERIALIZATION MODE")
    
class IntraDataset(Dataset):

    def __init__(self, file_list, root_dir, mode = "pre_order", p = None, val = False):
        
        
        #self.folder_path = folder_path
        #self.file_list = os.listdir(folder_path)  # Call os.listdir only once
        self.mode = mode
        self.root_dir = Path(root_dir)
        self.file_list = []
        self.val = val
        for rel_path in file_list:
            if p is not None:
                full_path = self.root_dir / f"p{p}" / rel_path
            else:
                full_path = self.root_dir / rel_path
            self.file_list.append(str(full_path))
        
        # Split dataset for train and validation
        total_files = len(self.file_list)
       
        print("Dataset size:", len(self))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        file_path = self.file_list[idx]

        # Use memory mapping to avoid loading full file into memory
        tree_data_np = np.load(file_path, mmap_mode='r')
        
        # Convert to tensor only when accessed
        tree_tensor = torch.tensor(tree_data_np, dtype=torch.float32)
        tree_tensor = tree_tensor.reshape((-1,39))

        if self.mode == "pre_order":

            serial_tree = list(tree_tensor.flatten().numpy())

            print(len(serial_tree))

            tree = deserialize(serial_tree, mode = "post_order")
            serial_tree = serialize_pre_order(tree, k=39)
            np_tree = np.array(serial_tree).reshape((-1,39))
            tree_tensor = torch.tensor(np_tree, dtype = torch.float32)

            
        if not self.val:
            return tree_tensor
        else:
            return tree_tensor, file_path


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
        
        # Save the model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, model_save_path)

    return best_loss

def save_best_model_gpt2(model, optimizer, epoch, loss, best_loss, model_save_path):
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
        
        # Save the model
        model.save_pretrained(model_save_path)
        
    return best_loss

import os

def erase_all_files(folder_path):

    # Iterate through all items in the folder

    for filename in os.listdir(folder_path):

        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path): os.remove(file_path)
