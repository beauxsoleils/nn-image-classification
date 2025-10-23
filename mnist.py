#!/usr/bin/env python3

"""
Contains the logic for pulling datasets from huggingface.

"""

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

def get_dataloaders_mnist(batch_size: int) -> [DataLoader, DataLoader]:
    """ 
    Saves to /data locally. 
    Returns two DataLoader objects.
    """

    # Convert the data to a PyTorch tensor and normalize it. 
    # Normalization is a common step for training neural networks.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, ), 
            (0.5, )
        )
    ])

    # Download the training dataset to current working directory.
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )

    # Download the testing dataset.
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )

    # Initialize DataLoaders with training and testing sets.
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )

    return train_loader, test_loader
    
if __name__ == "__main__": pass



