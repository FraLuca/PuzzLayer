import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Build a model with 3 layer for a classification of 10 classes
class CustomModel(nn.Module):
    def __init__(self, num_classes=10, input_channel=3):
        super(CustomModel, self).__init__()
        # 3 layer with batch normalization and activation function ReLU
        self.module = nn.Sequential(
            nn.Linear(input_channel*32*32, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.module(x)
        return x
    

class CustomModelMnist(nn.Module):
    def __init__(self, num_classes=10, input_channel=1):
        super(CustomModelMnist, self).__init__()
        # 3 layer with batch normalization and activation function ReLU
        self.module = nn.Sequential(
            nn.Linear(input_channel*28*28, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.module(x)
        return x
    
#############################################################################
# these are the models we are trying to reconstruct

class NND_mnist(nn.Module):
    def __init__(self, num_classes=10, input_channel=1, custom_init=False):
        super(NND_mnist, self).__init__()
        # 3 layer with batch normalization and activation function ReLU
        self.module = nn.Sequential(
            nn.Linear(input_channel*28*28, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, num_classes)
        )
        if custom_init:
            print("CUSTOM INITIALIZATION OF MLP")
            self.init_weights()

    def forward(self, x):
        x = x.view(x.size(0), -1) # flatten to [batch_size, -1]
        x = self.module(x)
        return x

    def init_weights(self):
        for name, p in self.module.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(p.data, nonlinearity="relu")
                # nn.init.normal_(p.data, std= 0.005 * (math.sqrt(2) / math.sqrt(p.data.shape[1])))
            elif "bias" in name:
                p.data.fill_(0)
        return


class MLP_mnist_4(nn.Module):
    def __init__(self, num_classes=10, input_channel=1):
        super(MLP_mnist_4, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(input_channel*28*28, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 25),
            nn.ReLU(),
            nn.Linear(25, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.module(x)
        return x


class MLP_mnist_7(nn.Module):
    def __init__(self, num_classes=10, input_channel=1):
        super(MLP_mnist_7, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(input_channel*28*28, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.module(x)
        return x
    

if __name__ == '__main__':
    # Test the model
    model = NND_mnist()
    print('Number of parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))