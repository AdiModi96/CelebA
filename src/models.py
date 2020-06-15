import torch
import torch.nn as nn

class CNN(nn.Module):
    input_channels = 1

    def __init__(self):
        super().__init__()

        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=CNN.input_channels, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='zeros', bias=True),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='zeros', bias=True),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='zeros', bias=True),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, padding_mode='zeros', bias=True),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, padding_mode='zeros', bias=True),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, padding_mode='zeros', bias=True),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2),

                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, padding_mode='zeros', bias=True),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, padding_mode='zeros', bias=True),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, padding_mode='zeros', bias=True),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, padding_mode='zeros', bias=True),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
        )

    def __repr__(self):
        return 'ResNet50'

    def forward(self, tensor):
        pass
