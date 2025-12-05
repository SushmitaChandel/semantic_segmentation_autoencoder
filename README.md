# Brief Description

This library implements an autoencoder model suggested in the paper “SAR Image Semantic Segmentation of Typical Oceanic and Atmospheric Phenomena” by Quankun Li. 

# Source of Dataset

The dataset can be downloaded from the link “https://drive.google.com/drive/folders/1rk8jwfWpJAQymbT35wJWMY-1KvawosOV?usp=sharing” and is the one detailed in the above paper. It is a semantic segmentation dataset containing 10 ocean features.

# Training

The train.py file needs to be run in order to train the model. Following is an example of a way to
run the file:

# Train from beginning using command lines:

python3 train.py 100 /Users/sushmitachandel/Desktop/dataset/ocean_features_pixel_level/archive/train/input/ /Users/sushmitachandel/Desktop/dataset/ocean_features_pixel_level/archive/val/input/ /Users/sushmitachandel/Desktop/dataset/ocean_features_pixel_level/archive/train/output/ /Users/sushmitachandel/Desktop/dataset/ocean_features_pixel_level/archive/val/output/ mps --resume_train False --save_dir train3 --batch_size 128

# Resume training using command lines

python3 train.py 100 /Users/sushmitachandel/Desktop/dataset/ocean_features_pixel_level/archive/train/input/ /Users/sushmitachandel/Desktop/dataset/ocean_features_pixel_level/archive/val/input/ /Users/sushmitachandel/Desktop/dataset/ocean_features_pixel_level/archive/train/output/ /Users/sushmitachandel/Desktop/dataset/ocean_features_pixel_level/archive/val/output/ mps --resume_train True --save_dir train3 --batch_size 128

50 denotes the number of epochs. 

Here, /Users/sushmitachandel/Desktop/dataset/ocean_features_pixel_level/archive/train/input/ is the absolute path containing training images

/Users/sushmitachandel/Desktop/dataset/ocean_features_pixel_level/archive/val/input/  is the absolute path containing validation images

/Users/sushmitachandel/Desktop/dataset/ocean_features_pixel_level/archive/train/output/  is the absolute path containing training ground truths

/Users/sushmitachandel/Desktop/dataset/ocean_features_pixel_level/archive/val/output/ is the absolute path containing validation ground truths

mps denotes that kind of compute being used is metal performance shaders. Type gpu if you are using Nvidia GPU and type cpu if you do not.

train1 denotes the name of the folder where the model and curves would be saved. Please change the name on a new run except when you are not continuing an old training

# Testing

The test.ipynb file in the tests folder can be run for evaluating the trained model on a sample images. 

# Folder Structure

```
Following is a brief about the folder structure:
|- src/
| 	|- data_loader.py # Data loading and preprocessing functions.
| 	|- models_src.py # Model Architecture
| 	|- utils.py # Utility functions
| 	|- train.py # Training loop and logic
|
|-models/
| 	|-train # Folder containing trials done. It contains the .pth files having saved models and
checkpoints.
|
|-tests/
| 	|-test.ipynb # Unit tests for the code.
|
|-README # Project overview and instructions.
|
|-Requirements # It details the library and python versions that were used in this project.
|
|-data # Contains sample images and ground truths needed during testing
```


# Requirements

Important requirements are detailed below. The code was validated using the following:
Python - 3.11.8
PyTorch - 2.5.1
Torchvision - 0.20.1
