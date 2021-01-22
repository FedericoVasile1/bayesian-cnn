import torch.nn as nn

from src.utils.model import GlobalAvgPool

class MAlexNet(nn.Module):
    def __init__(self, num_classes, in_channels, drop=0.5, activation_function='relu'):
        super(MAlexNet, self).__init__()

        if activation_function == 'softplus':
            self.act = nn.Softplus
        elif activation_function == 'relu':
            self.act = nn.ReLU
        elif activation_function == 'tanh':
            self.act = nn.Tanh
        else:
            raise ValueError("Only softplus, relu or tanh supported")

        self.features = nn.Sequential(
            nn.Dropout(drop),
            nn.Conv2d(in_channels, 64, 11, stride=4, padding=5, bias=True),
            self.act(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(drop),
            nn.Conv2d(64, 192, 5, padding=2, bias=True),
            self.act(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(drop),
            nn.Conv2d(192, 384, 3, padding=1, bias=True),
            self.act(),
            nn.Dropout(),
            nn.Conv2d(384, 256, 3, padding=1, bias=True),
            self.act(),
            nn.Dropout(),
            nn.Conv2d(256, 128, 3, padding=1, bias=True),
            self.act(),
            GlobalAvgPool(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(128, num_classes, bias=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x