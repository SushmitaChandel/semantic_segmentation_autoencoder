#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 21:03:26 2025

@author: sushmitachandel
"""

import torch
import torch.nn as nn

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            
            # Block 1: (1, x, y) -> (32, x/2, y/2)
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 2: (32, x/2, y/2) -> (64, x/4, y/4)
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 3: (64, x/4, y/4) -> (128, x/8, y/8)
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

             # Bottleneck: (128, x/8, y/8) -> (256, x/8, y/8)
            nn.Conv2d(128, 256, 3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU()
        )

        # Decoder with ConvTranspose2D for upsampling and skip connection emulation
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # (256, a, b) -> (128, 2a, 2b)
        self.dropout = nn.Dropout2d(0.5)

        self.decoder = nn.Sequential( # (256, c, d) -> (128, c, d)
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(128, 11, 1)  # (128, e, f) -> (11, e, f)
        )
        

    def forward(self, x):
        # Encoder forward
        e1 = self.encoder[:6](x)        # just before maxpool1 e1 shape (32, x, y)
        e2 = self.encoder[6:13](e1)     # just before maxpool2 output e2 shape (64, x/2, y/2)
        e3 = self.encoder[13:20](e2)    # just before maxpool3 output e3 shape (128, x/4, y/4)
        encoded = self.encoder[20:](e3) # bottleneck shape (256, x/8, y/8)
        
        # Decoder forward
        up1 = self.upconv1(encoded)     # up1 shape (128, x/4, x/4)
        drop1 = self.dropout(e3)        # drop1 shape (128, x/4, x/4)
        concat = torch.cat((up1, drop1), dim=1)  # concat along channel axis -> 256 channels (256, x/4, x/4)
        
        x = self.decoder(concat) # output x shape (11, x/4, x/4)
        return x      
    
# Give some initial weights
def init_weights(m):
    if hasattr(m, 'weight') and m.weight is not None:
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    if hasattr(m, 'bias') and m.bias is not None:
        torch.nn.init.zeros_(m.bias)

def create_model(device='mps'):
    
    model = Autoencoder()
    model.apply(init_weights)
    model = model.to(device)
    
    return model   

# def test():
    
#     # Run this to test weather everything is ok or not!
        
#     x = torch.randn(2, 1, 256, 256)
#     model = Autoencoder()
#     x = model.forward(x)
    
# test()



