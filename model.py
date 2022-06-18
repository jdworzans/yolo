from enum import Flag
from turtle import forward
import pytorch_lightning as pl
from torch import nn
import torch

class ConvBlock(nn.Module):
    def __init__(self, in_chanels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_chanels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))


class InitialConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(2, 2),
            
            ConvBlock(64, 192, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),

            ConvBlock(192, 128, kernel_size=1),
            ConvBlock(128, 256, kernel_size=3),
            ConvBlock(256, 256, kernel_size=1),
            ConvBlock(256, 512, kernel_size=3),
            nn.MaxPool2d(2, 2),

            nn.Sequential(
                *[
                    nn.Sequential(ConvBlock(512, 256, kernel_size=1), ConvBlock(256, 512, kernel_size=3, padding=1))
                    for _ in range(4)
                ]
            ),
            ConvBlock(512, 512, kernel_size=1),
            ConvBlock(512, 1024, kernel_size=1, padding=1),
            nn.MaxPool2d(2, 2),

            nn.Sequential(
                *[
                    nn.Sequential(ConvBlock(1024, 512, kernel_size=1), ConvBlock(512, 1024, kernel_size=3, padding=1))
                    for _ in range(2)
                ]
            )
        )

    def forward(self, x):
        return self.conv(x)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = InitialConv()
        self.fc = nn.Sequential(nn.AvgPool2d(7), nn.Flatten(), nn.Linear(1024, 1000))

    def forward(self, x):
        return self.fc(self.conv(x))

class YOLO(nn.Module):
    def __init__(self, n_features=7, n_bboxes=2, n_classes=20):
        super().__init__()
        self.S = n_features
        self.B = n_bboxes
        self.C = n_classes
        self.initial_conv = InitialConv()
        self.final_conv = nn.Sequential(
            ConvBlock(1024, 1024, kernel_size=3, padding=1),
            ConvBlock(1024, 1024, kernel_size=3, stride=2, padding=1),
            ConvBlock(1024, 1024, kernel_size=3, padding=1),
            ConvBlock(1024, 1024, kernel_size=3, padding=1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.S * self.S * 1024, 496), nn.LeakyReLU(0.1, inplace=True), nn.Dropout(0.5),
            nn.Linear(496, self.S * self.S * (5 * self.B + self.C)), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.fc(self.final_conv(self.initial_conv(x))).view(-1, self.S, self.S, 5 * self.B + self.C)

if __name__ == "__main__":
    example = torch.randn(10, 3, 448, 448)
    a = YOLO()
    print(example.shape, a(example).shape)
