import math
import torch.nn as nn
from layers import BBB_Linear, BBB_Conv2d
from layers import BBB_LRT_Linear, BBB_LRT_Conv2d
from layers import FlattenLayer, ModuleWrapper

class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        return x.mean(dim=(2, 3))

class BBB3Conv3FC(ModuleWrapper):
    """

    Simple Neural Network having 3 Convolution
    and 3 FC layers with Bayesian layers.
    """
    def __init__(self, outputs, inputs, priors, layer_type='lrt', activation_type='softplus'):
        super(BBB3Conv3FC, self).__init__()

        self.num_classes = outputs
        self.layer_type = layer_type
        self.priors = priors

        if layer_type=='lrt':
            BBBLinear = BBB_LRT_Linear
            BBBConv2d = BBB_LRT_Conv2d
        elif layer_type=='bbb':
            BBBLinear = BBB_Linear
            BBBConv2d = BBB_Conv2d
        else:
            raise ValueError("Undefined layer_type")
        
        if activation_type=='softplus':
            self.act = nn.Softplus
        elif activation_type=='relu':
            self.act = nn.ReLU
        elif activation_type=='tanh':
            self.act = nn.Tanh
        else:
            raise ValueError("Only softplus,relu and tanh supported")

        self.conv1 = BBBConv2d(inputs, 32, 5, padding=2, bias=True, priors=self.priors)
        self.act1 = self.act()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = BBBConv2d(32, 64, 5, padding=2, bias=True, priors=self.priors)
        self.act2 = self.act()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = BBBConv2d(64, 128, 5, padding=1, bias=True, priors=self.priors)
        self.act3 = self.act()
        #self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.pool3 = GlobalAvgPool()

        self.flatten = FlattenLayer(128)
        self.fc1 = BBBLinear(128, 256, bias=True, priors=self.priors)
        self.act4 = self.act()

        self.fc2 = BBBLinear(256, 256, bias=True, priors=self.priors)
        self.act5 = self.act()

        self.fc3 = BBBLinear(256, outputs, bias=True, priors=self.priors)