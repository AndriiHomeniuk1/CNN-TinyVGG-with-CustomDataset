## Training Custom Model on Horse or Human Dataset using TinyVGG Architecture

This repository contains code to train a custom convolutional neural network (CNN) model on the "Horse or Human" dataset using the TinyVGG architecture. The TinyVGG model is a simplified version of the VGG (Visual Geometry Group) architecture designed for small-scale image classification tasks.

### Dataset

The "Horse or Human" dataset consists of images of horses and humans, segmented into training and testing sets. The dataset is preprocessed and loaded using PyTorch's `ImageFolder` and custom `Dataset` classes for efficient handling of image data.

### Model Architecture

The TinyVGG model used in this project consists of two convolutional blocks followed by a fully connected classifier. Each convolutional block includes two convolutional layers with ReLU activations and max-pooling layers for feature extraction.

### Training

The model is trained using PyTorch's capabilities for defining custom training loops. The training process includes optimizing the model parameters using the Adam optimizer and computing the categorical cross-entropy loss between predicted and true labels.

### Evaluation

After training for a specified number of epochs, the model's performance is evaluated on a separate test set to measure its accuracy and loss metrics. These metrics are plotted to visualize the training progress and model performance over epochs.

### Prediction

The trained model is capable of making predictions on new images. A demonstration of model prediction on custom images is provided, showing how the model identifies and classifies objects as either a horse or a human.

## Results

### Training Results

- **Training Loss**:
  - Epoch 1: 0.6656
  - Epoch 2: 0.5596
  - Epoch 3: 0.4771

- **Training Accuracy**:
  - Epoch 1: 56.19%
  - Epoch 2: 71.69%
  - Epoch 3: 76.52%

### Test Results

- **Test Loss**:
  - Epoch 1: 0.5791
  - Epoch 2: 0.4156
  - Epoch 3: 0.3709

- **Test Accuracy**:
  - Epoch 1: 67.58%
  - Epoch 2: 83.20%
  - Epoch 3: 87.11%

## Repository Structure

- `README.md`: This file provides an overview of the project.
- `TinyVgg_with_custom_dataset.ipynb`: Jupyter notebook containing the code for training the model.
- `saved_models`: Directory where the trained model is saved.
