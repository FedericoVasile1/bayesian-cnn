from lib.PyTorchBayesianCNN.layers import ModuleWrapper


class GlobalAvgPool(ModuleWrapper):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

    def forward(self, x):
        return x.mean(dim=(2, 3))