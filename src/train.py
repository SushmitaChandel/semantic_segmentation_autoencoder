#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 11:18:20 2025

@author: sushmitachandel
"""

import argparse
import os
import sys
import torch.optim as optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output
import torch.nn.functional as F
from torchmetrics.segmentation import MeanIoU, DiceScore

import data_loader
import models_src
import utils

# Performs one_hot_encoding for tensor containing probabilities of the form (N, C, H, W)
def one_hot_encoded(probs):
    # Step 1: Get predicted class indices (N, C, H, W)
    class_indices = probs.argmax(dim=1)  # class with max prob per pixel
    # Step 2: One-hot encode the class indices to (N, H, W, C)
    one_hot = F.one_hot(class_indices, num_classes=probs.shape[1])
    # Step 3: Permute to (N, C, H, W) to match original tensor shape
    one_hot = one_hot.permute(0, 3, 1, 2).float()
    return one_hot

def train(n_epochs, train_loader, val_loader, classes, device, save_dir="checkpoints"):
    
    num_classes = len(classes)
    # print(f'no of class {num_classes}')
    
    model = models_src.create_model(device=device)
    
    # Define Loss Function
    criterion_name = 'CrossEntropyLoss' 
    criterion = nn.CrossEntropyLoss()
    
    # Define Optimizer 
    optimizer_name = 'Adam'
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    
    # Initialize training losses and dice lists
    os.makedirs(save_dir, exist_ok=True)
    # print(f'I AM HERE {save_dir}')
    # Check whether the best epoc is saved or not.
    path_last = save_dir+"/lastepoch.pth"
    path_best = save_dir+"/bestepoch.pth"
    if os.path.exists(path_best):
        print('EXISTS')
        checkpoint_best = torch.load(path_best, map_location=device)
        best_val_loss = checkpoint_best['best_val_loss']
        best_epoch = checkpoint_best['best_epoch']
        checkpoint_last = torch.load(path_last, map_location=device)
        model.load_state_dict(checkpoint_last['model_state_dict'])
        optimizer.load_state_dict(checkpoint_last['optimizer_state_dict'])
        train_losses_epoch = checkpoint_last['train_losses_epoch']
        val_losses_epoch  = checkpoint_last['val_losses_epoch']
        train_dice_epoch = checkpoint_last['train_dice_epoch']
        val_dice_epoch  = checkpoint_last['val_dice_epoch']
    else:
        best_val_loss = float('inf') 
        best_epoch = 0
        train_losses_epoch = [] # Trackers - accumulated over epochs
        val_losses_epoch= []
        train_dice_epoch = []
        val_dice_epoch = []
        
    len_train_dataset = len(train_loader.dataset)
    len_val_dataset = len(val_loader.dataset)
    
    # Begin training
    start_epoch = len(train_losses_epoch)
    for epoch in range(start_epoch, start_epoch+n_epochs):
        
        epoch = epoch+1
        print(f'Starting Epoch: {epoch}...')
        
        ######### Training #########
        running_loss = 0
        dice_score = 0
        for i, data in enumerate(train_loader, 0):

            i += 1
            
            X_train, y_train = data
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            # y_train = y_train.permute(0, 3, 1, 2)
            
            # Predict and collect all the correct predictions
            y_pred = model(X_train)
            # print(f'y_pred shape {y_pred.shape}')
            y_pred_softmax = F.softmax(y_pred, dim=1)
            y_pred_onehot = one_hot_encoded(y_pred_softmax)
            ds = DiceScore(num_classes=num_classes, average="micro")
            dice_score += ((ds(y_pred_onehot, y_train)).item())* X_train.size(0)
            
            # Compute loss
            loss = criterion(y_pred, y_train)
            # print(f'loss is {loss}')
            running_loss += loss.item() * X_train.size(0) 
            # print(f'running_loss is {running_loss}')
            
            # Clear the gradients before training by setting to zero
            # Required for a fresh start
            optimizer.zero_grad()
            
            # Backpropogate
            loss.backward()

            # Update weights
            optimizer.step()         
            
            # if i % 5 == 0: # show interim results every 50 mini-batches
            #     # print(f'running_loss is: {running_loss} and number of items are : {X_train.size(0)*i}')
            #     l = running_loss / (X_train.size(0)*i)
            #     print(f'Epoch: {epoch}, Mini-Batches Completed: {(i)}, Loss: {l:.3f}')
            
        epoch_loss = running_loss / len_train_dataset
        train_losses_epoch.append(epoch_loss)
        epoch_dice_score = dice_score / len_train_dataset
        train_dice_epoch.append(epoch_dice_score)
        
        # Evaluate model on val/test dataset
        running_loss = 0
        dice_score = 0
        with torch.no_grad():
            for j, data in enumerate(val_loader, 0):
                X_val, y_val = data
                X_val = X_val.to(device)
                y_val = y_val.to(device)
                # y_val = y_val.permute(0, 3, 1, 2)
                # print(f'y_val shape {y_val.shape}')

                # Predict and collect all the correct predictions
                y_pred = model(X_val)
                # print(f'y_pred shape {y_pred.shape}')
                loss = criterion(y_pred, y_val) # entropy loss takes softmax inherently
                running_loss += loss.item() * X_val.size(0)

                y_pred_softmax = F.softmax(y_pred, dim=1)
                y_pred_onehot = one_hot_encoded(y_pred_softmax)
                ds = DiceScore(num_classes=num_classes, average="micro")
                dice_score += ((ds(y_pred_onehot, y_val)).item())* X_train.size(0)
        
        epoch_loss = running_loss / len_val_dataset
        val_losses_epoch.append(epoch_loss)
        epoch_dice_score = dice_score / len_val_dataset
        val_dice_epoch.append(epoch_dice_score)
        
        ########### Saving best results #########
        if epoch_loss < best_val_loss:
            print(f'Saving best epoch')
            best_val_loss = epoch_loss
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'criterion_name': criterion_name,
                'optimizer_name': optimizer_name,
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'best_val_dice': epoch_dice_score,
            }, f"{save_dir}/bestepoch.pth")
            print(f"Saved best model at epoch {epoch},"
                  f" with val_loss: {best_val_loss:.4f}")
            
        ########### Saving last epoch #########
        # Save after each 2nd epoch
        print(f'Saving last epoch')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses_epoch': train_losses_epoch,
            'val_losses_epoch': val_losses_epoch,
            'val_dice_epoch': val_dice_epoch,
            'train_dice_epoch': train_dice_epoch,
            'criterion_name': criterion_name,
            'optimizer_name': optimizer_name,
            'last_epoch': epoch,
        }, f"{save_dir}/lastepoch.pth")
        
        # ########## Plotting on the go #########
        epochs_list = list(range(1, len(train_losses_epoch)+1))
        # get_plot(epochs_list, train_losses_epoch, val_losses_epoch, 
        #          "Epoch", "Loss", "Training & Validation Loss")
        # clear_output(wait=True)
            
        # get_plot(epochs_list, train_dice_epoch, val_dice_epoch, 
        #             "Epoch", "Accuracy", "Training & Validation dice")
        # clear_output(wait=True)
        if epoch % 10 == 0:
            print(f'Epochs till now: {epochs_list[-10:]}')
            print(f'Training metrics for last few epochs till now: {train_dice_epoch[-10:]}')
            print(f'Validation metrics till now: {val_dice_epoch[-10:]}')

    print('Finished Training')
    
    
def main(n_epochs=20, train_path_input="train_input/", val_path_input="val_input/",
         train_path_output="train_output/", val_path_output="val_output/", 
         device="mps", resume_train="False", save_dir="train2", batch_size=128):
    
    train_loader, val_loader, classes = data_loader.get_dataloaders(train_path_input,
                                    val_path_input,train_path_output,val_path_output,
                                    batch_size=batch_size)
    print(f'The classes are: {classes}')
    if resume_train == "False":
        if os.path.isdir(save_dir):
            print(f"Error: The save_dir '{save_dir}' already exists. Please specify a different directory.")
            sys.exit(1)
        else:
            print("Folder does not exisxt. Creating the folder.")
            
    if resume_train == "True":
        if os.path.isdir(save_dir) == False:
            print(f"Error: The save_dir '{save_dir}' does not exists. Please specify a different directory.")
            sys.exit(1)
        else:
            print("Continuing training.")
    
    train(n_epochs, train_loader, val_loader, classes, device, save_dir=save_dir)
    
    utils.save_curves(device, save_dir=save_dir)
    
    path_last = save_dir+"/lastepoch.pth"
    path_best = save_dir+"/bestepoch.pth"
    checkpoint_best = torch.load(path_best, map_location=device)
    checkpoint_last = torch.load(path_last, map_location=device)
    train_losses_epoch = checkpoint_last['train_losses_epoch']
    val_losses_epoch  = checkpoint_last['val_losses_epoch']
    train_dice_epoch = checkpoint_last['train_dice_epoch']
    val_dice_epoch  = checkpoint_last['val_dice_epoch']
    best_val_loss = checkpoint_best['best_val_loss']
    best_val_dice = checkpoint_best['best_val_dice']
    path_file =  save_dir+"/training_log.txt"
    total_epochs = len(train_losses_epoch)
    utils.save_training_log(path_file, best_val_dice, best_val_loss, total_epochs, 
                            train_losses_epoch, val_losses_epoch, 
                             train_dice_epoch, val_dice_epoch)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run train with command line arguments")
    parser.add_argument("n_epochs", type=int, help="Number of epochs for training")
    parser.add_argument("train_path_input", type=str, help="Path to train dataset directory")
    parser.add_argument("val_path_input", type=str, help="Path to val dataset directory")
    parser.add_argument("train_path_output", type=str, help="Path to ground truths for train dataset directory")
    parser.add_argument("val_path_output", type=str, help="Path to ground truths for val dataset directory")
    parser.add_argument("device", type=str, help="Device to use for training (e.g., 'cpu' or 'cuda' or 'mps')")
    parser.add_argument("--resume_train", type=str, default="False", help="Flag suggesting whether or not to resume training from the previous step")
    parser.add_argument("--save_dir", type=str, default="train2", help="Directory to save outputs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for data loaders")
    args = parser.parse_args()
    
    n_epochs = args.n_epochs
    device = args.device
    resume_train = args.resume_train
    save_dir = args.save_dir
    batch_size = args.batch_size
    train_path_input = args.train_path_input
    val_path_input = args.val_path_input
    train_path_output = args.train_path_output
    val_path_output = args.val_path_output
    
    # Get absolute path of the directory to save all training history.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "../models/"+save_dir)
    save_dir = os.path.abspath(models_dir)
    # print(f'I AM HERE {save_dir}')
    
    main(n_epochs=n_epochs, train_path_input=train_path_input, val_path_input=val_path_input,
        train_path_output=train_path_output, val_path_output=val_path_output, device=device, 
        resume_train=resume_train, save_dir=save_dir)
    
            
        
    
    
    