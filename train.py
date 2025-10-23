#!/usr/bin/env python3

"""
Handles all the training logic. Defined manually.
"""

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from model import Model
from typing import Optional
from tqdm import tqdm

def train(
    model: nn.Module, 
    optimizer: optim.Optimizer, 
    loss_fn: Optional,
    dataloader: DataLoader,
    device: str
) -> float:
    """
    Handles the logic required to train a neural network.
    Backpropagation is our training method here.
    It is used along with evaluate() in the full training loop.
    """

    model.train()
    model = model.to(device)
    total_loss = 0.0

    for feature, target in tqdm(dataloader):
        # load to device
        feature, target = feature.to(device), target.to(device)
        # zero gradients from previous step
        optimizer.zero_grad()            
        # forward pass
        preds = model(feature)                
        # compute loss
        loss = loss_fn(preds, target)        
        # backpropagation step: compute gradients
        loss.backward()                  
        # update parameters using gradients
        optimizer.step()                 

        total_loss += loss.item() * feature.size(0)

    print("Epoch complete!")
    return total_loss / feature.size(0)

def evaluate(
    model: nn.Module, 
    loss_fn: Optional,
    dataloader: DataLoader,
    device: str
) -> float:
    """
    Evaluates and retuns the loss after each training epoch.
    """

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            total_loss += loss.item() * xb.size(0)

    return total_loss / len(dataloader.dataset)
    
def run_training_loop(
    model: nn.Module,
    loss_fn: Optional,                  
    optimizer: optim.Optimizer,
    training_loader: DataLoader,
    testing_loader: DataLoader,
    device: str,
    epochs: int,
) -> float:
    """
    Run the full training loop. returns the final training loss as a float int.
    """ 

    model.to(device)

    for epoch in range(1, epochs + 1):
        train_loss: float = train(
            model, 
            optimizer, 
            loss_fn, 
            training_loader, 
            device
        )
        print("Evaluating...")
        val_loss: float = evaluate(
            model, 
            loss_fn, 
            testing_loader,
            device
        )
        print(f"Epoch: {epoch:03d}  Training loss: {train_loss:.6f}  Evaluation loss: {val_loss:.6f}\n")
    return train_loss

if __name__ == "__main__": pass




