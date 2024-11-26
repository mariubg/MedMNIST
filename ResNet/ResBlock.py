import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):

        #Non-linear layers
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding, stride),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x):
        x_layers = self.layers.forward(x)
        x_skip = self.skip_connection.forward(x)
        output = x_layers + x_skip
        return output