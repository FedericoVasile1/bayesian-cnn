import torch.nn as nn

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #nn.init.xavier_uniform(m.weight, gain=np.sqrt(2))
        nn.init.normal_(m.weight, mean=0, std=1)
        nn.init.constant(m.bias, 0)

class M3Conv3FC(nn.Module):
    def __init__(self, num_classes, in_channels, drop=0.5, activation_function='relu'):
        super(M3Conv3FC, self).__init__()

        if activation_function == 'relu':
            self.act = nn.ReLU
        elif activation_function == 'tanh':
            self.act = nn.Tanh
        elif activation_function == 'softplus':
            self.act = nn.Softplus
        else:
            raise ValueError('Unknow '+activation_function+' activation function. '
                            'Only softplus, relu and tanh supported.')

        self.features = nn.Sequential(
            nn.Dropout(drop),
            nn.Conv2d(in_channels, 32, 5, stride=1, padding=2),
            self.act(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(drop),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            self.act(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(drop),
            nn.Conv2d(64, 128, 5, stride=1, padding=1),
            self.act(),
            #nn.MaxPool2d(kernel_size=3, stride=2),      # replace this with global avg pool
        )
        self.classifier = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(128, 256),
            self.act(),
            nn.Dropout(drop),
            nn.Linear(256, 256),
            self.act(),
            nn.Dropout(drop),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=(2, 3))  # global average pooling
        x = self.classifier(x)
        return x
