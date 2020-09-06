import torch.nn as nn

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #nn.init.xavier_uniform(m.weight, gain=np.sqrt(2))
        nn.init.normal_(m.weight, mean=0, std=1)
        nn.init.constant(m.bias, 0)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.flatten(start_dim=1)

class Dropout3Conv3FC(nn.Module):
    def __init__(self, outputs, inputs, drop=0.5):
        super(Dropout3Conv3FC, self).__init__()
        self.features = nn.Sequential(
            nn.Dropout(drop),
            nn.Conv2d(inputs, 32, 5, stride=1, padding=2),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(drop),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(drop),
            nn.Conv2d(64, 128, 5, stride=1, padding=1),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            Flatten(),
            nn.Dropout(drop),
            nn.Linear(2 * 2 * 128, 1000),
            nn.Softplus(),
            nn.Dropout(drop),
            nn.Linear(1000, 1000),
            nn.Softplus(),
            nn.Dropout(drop),
            nn.Linear(1000, outputs)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
