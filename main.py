#!/usr/bin/env python3

"""
The main() script for initializing our model and running training epochs.

"""

import torch, utils

from torch import optim, nn
from model import Model
from mnist import get_dataloaders_mnist
from train import run_training_loop, evaluate
from args import build_interface


def main():
    
    # Build ui
    args: ArgumentParser = build_interface()

    # Select cuda for computation
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    # Prepare datasets
    training_loader, testing_loader = get_dataloaders_mnist(
        batch_size=16
    )

    # Define model
    model: nn.Module = Model()

    # Initialize loss function
    loss_fn = nn.CrossEntropyLoss()

    # Initialize optimizer
    optimizer = optim.SGD(
        params=model.parameters(), 
        lr=args.learning_rate
    )

    # Begin training 
    print("Training begins!\n")
    final_train_loss: float = run_training_loop(
        model=model, 
        loss_fn=loss_fn, 
        optimizer=optimizer, 
        training_loader=training_loader, 
        testing_loader=testing_loader,
        device=device,
        epochs=args.epochs
    )

    print(f"Final training loss: {final_train_loss}\n")
    
    if args.save:
        checkpoint_path = utils.save_model_state(model)
        print(f" Model weights saved to /{checkpoint_path}\n")


if __name__ == "__main__": main()



