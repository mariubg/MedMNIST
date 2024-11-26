import torch
import torch.nn as nn
from ResNet.ResBlock3D import ResBlock3D

class ResNet3D(nn.Module):
    def __init__(self, layers, num_classes=1000, input_channels=1):  # Default input_channels=1 for most medical imaging
        super().__init__()
        self.layers = layers
        self.in_channels = 64
        
        # Change first conv layer to 3D
        self.conv1 = nn.Conv3d(input_channels, self.in_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(self.in_channels)  # Change to BatchNorm3d
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)  # Change to MaxPool3d
        
        # Use ResBlock3D instead of ResBlock
        self.res_layer1 = self.layer(64, 64, self.layers[0])
        self.res_layer2 = self.layer(64, 128, self.layers[1], stride=2)
        self.res_layer3 = self.layer(128, 256, self.layers[2], stride=2)
        self.res_layer4 = self.layer(256, 512, self.layers[3], stride=2)
        
        self.avg = nn.AdaptiveAvgPool3d((1, 1, 1))  # Change to AdaptiveAvgPool3d
        self.fc = nn.Linear(512, num_classes)

    def layer(self, in_channels, out_channels, nums, stride=1):
        layers = []
        layers.append(ResBlock3D(in_channels, out_channels, stride=stride))
        for i in range(1, nums):
            layers.append(ResBlock3D(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.res_layer1(x)
        x = self.res_layer2(x)
        x = self.res_layer3(x)
        x = self.res_layer4(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
