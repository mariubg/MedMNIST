import torch
import torch.nn as nn
from ResBlock import ResBlock

class ResNet(nn.Module):
    def __init__(self, layers, num_classes = 1000):
        super().__init__()
        self.layers = layers
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res_layers = []
        for i in range(len(self.layers)):
            if i == 0:
                stride = 1
            else:
                stride = 2
            self.res_layers.append(self.layer(64*2**i, self.layers[i], stride=stride))

        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)

    def layer(self, in_channels, out_channels, nums, stride = 1):
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride=stride))
        for i in range(1, nums):
            layers.append(ResBlock(out_channels, out_channels, stride=stride))
        return nn.Sequential(*layers)
    

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

