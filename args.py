#!/usr/bin/env python3

import argparse 

def build_interface():

    parser = argparse.ArgumentParser(
        prog="Image Classification",
        description="Defines and trains a forward feed neural network. Returns weights to disc.",
        allow_abbrev=True
    )

    parser.add_argument(
        '--epochs', 
        type=int, 
        default=250, 
        help='Training epochs ~ default: 20'
    )

    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=16, 
        help='Dataset batch size ~ default: 16'
    )

    parser.add_argument(
        '--learning_rate', 
        type=float, 
        default=0.01, 
        help='Learning rate for the optimizer ~ default: 0.01'
    )
    
    parser.add_argument(
        '--save',
        type=bool,
        default=True,
        help='Save model weights to local directory'
    )

    return parser.parse_args()

if __name__ == '__main__': pass



