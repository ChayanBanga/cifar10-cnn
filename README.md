# CIFAR-10 Image Classification with CNN & ResNet
> Built from scratch using PyTorch — exploring optimization techniques to push from **75.81% → 88.52%** test accuracy.

---

## Overview

This project implements a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset. Starting from a basic 3-layer CNN, I progressively applied optimization techniques to improve performance, culminating in a ResNet-style architecture with skip connections.

**Final Result: 88.52% Test Accuracy**

---

## Dataset

**CIFAR-10** — 60,000 color images (32×32) across 10 classes:

| Label | Class |
|---|---|
| 0 | Airplane |
| 1 | Automobile |
| 2 | Bird |
| 3 | Cat |
| 4 | Deer |
| 5 | Dog |
| 6 | Frog |
| 7 | Horse |
| 8 | Ship |
| 9 | Truck |

- 50,000 training images
- 10,000 test images

---

## Project Structure

```
cifar10-cnn/
├── train.py               ← full training script
├── app.py                 ← streamlit demo app
├── cifar10_resnet.pth     ← saved model weights
└── README.md
```

---

## Model Architecture

### Final Model — ResNet-style CNN

```
Input (3, 32, 32)
    ↓
Conv Block 1: Conv2d(3→32) → BN → ReLU → ResBlock(32) → MaxPool  →  (32, 16, 16)
    ↓
Conv Block 2: Conv2d(32→64) → BN → ReLU → ResBlock(64) → MaxPool  →  (64, 8, 8)
    ↓
Conv Block 3: Conv2d(64→128) → BN → ReLU → ResBlock(128) → MaxPool  →  (128, 4, 4)
    ↓
Conv Block 4: Conv2d(128→256) → BN → ReLU → ResBlock(256) → MaxPool  →  (256, 2, 2)
    ↓
Flatten → Linear(1024→256) → ReLU → Dropout(0.5) → Linear(256→10)
    ↓
Output (10 classes)
```

### ResBlock (Skip Connection)

```python
class ResBlock(nn.Module):
    def forward(self, x):
        out = self.block(x)
        out = out + x      # add original input back
        return self.relu(out)
```

The skip connection allows gradients to flow directly backwards, enabling deeper networks to train without vanishing gradient issues.

---

## Optimization Journey

| Model | Test Acc | Improvement | Notes |
|---|---|---|---|
| Base CNN — 3 blocks, 10 epochs | 75.81% | baseline | Simple Conv→ReLU→Pool |
| + Data Augmentation, 10 epochs | 76.19% | +0.38% | RandomFlip, RandomCrop |
| + Data Augmentation, 30 epochs | 80.37% | +4.56% | More training time |
| + Better Normalization | 80.67% | +0.30% | CIFAR-10 exact mean/std |
| + Batch Normalization | 81.00% | +0.33% | BN after every Conv layer |
| + Deeper CNN (4 blocks) | 84.23% | +3.23% | Added 128→256 conv block |
| + ResNet skip connections | **88.52%** | **+4.29%** | Residual connections |
| **Total improvement** | | **+12.71%** | |

---

## Techniques Used

### 1. Data Augmentation
Artificially increases training data variety, reduces overfitting:
```python
transforms.RandomHorizontalFlip()
transforms.RandomCrop(32, padding=4)
transforms.RandomRotation(10)
```

### 2. Batch Normalization
Normalizes activations after each conv layer — speeds up training and stabilizes learning:
```python
nn.BatchNorm2d(channels)
```

### 3. Dropout
Randomly zeros 50% of neurons during training — forces the model to not rely on any single neuron:
```python
nn.Dropout(0.5)
```

### 4. Deeper Architecture
Going from 3 → 4 conv blocks increased model capacity significantly (+3.23% accuracy).

### 5. Residual Connections (ResNet)
Skip connections solve the vanishing gradient problem, allowing deeper networks to train properly. Biggest single improvement: **+4.29%**.

---

## Results

```
Base Model Test Accuracy  :  75.81%
Final Model Test Accuracy :  88.52%
Total Improvement         : +12.71%
```

### Cat Image Confidence Progression
```
Batch Norm model  :  41.0%  → predicted wrong
Deeper CNN model  :  73.4%  → predicted correct
ResNet model      :  85%+   → predicted correct
```

---

## Setup & Usage

### Install dependencies
```bash
pip install torch torchvision tqdm streamlit
```

### Train the model
```bash
python train.py
```

### Run the Streamlit app
```bash
streamlit run app.py
```

---

## Requirements

```
torch
torchvision
tqdm
streamlit
Pillow
```

---

## Key Learnings

- Data augmentation hurts training accuracy but improves test accuracy — that's a good thing (less overfitting)
- Batch normalization is a free performance boost — always add it
- Model depth matters more than training longer after a certain point
- ResNet's skip connections are a simple idea with a massive impact
- Train/test accuracy gap is a direct measure of overfitting — smaller gap = better generalization

---

## Live Demo

🚀 [Hugging Face Spaces — Live Demo](#) *(link coming soon)*

---

## Author

Built as part of an ML internship task — Task 1: Image Classification with CNNs.
