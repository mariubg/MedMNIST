import torch
import torch.nn as nn

class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Change Conv2d to Conv3d
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm3d(out_channels),  # Change to BatchNorm3d
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm3d(out_channels),  # Change to BatchNorm3d
        )
        self.relu = nn.ReLU()
        
        # Modify skip connection for 3D
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)  # Change to BatchNorm3d
            )
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x):
        x_layers = self.layers.forward(x)
        x_skip = self.skip_connection.forward(x)
        output = self.relu(x_layers + x_skip)
        return output