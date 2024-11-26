import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

from ResNet.ResNet import ResNet
from medmnist import PathMNIST, Evaluator

# preprocessing
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

#Load data
train_dataset = PathMNIST(split='train', transform=data_transform, download=True, size=224, mmap_mode='r')
train_loader = data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)


model = ResNet([3, 4, 6, 3], 11)