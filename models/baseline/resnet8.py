import torch.nn as nn
import torch.nn.functional as F
from thop import profile
import torch

class ResNet8(nn.Module):
    def __init__(self, input_channels, out_classes, size=1):
        super(ResNet8, self).__init__()
        self.structures = {1: [16,16,32,64,64],
                          2: [10,10,26,42,42],
                          3: [6,8,24,32,32],
                          4: [5,7,18,32,32],
                          5: [3,6,16,32,32],
                          8: [2,4,12,24,16]}
        self.size = size
        self.input_channels = input_channels
        self.out_classes = out_classes
        self.set_size(size)

    def __set_structure(self):
        self.conv0 = nn.Conv2d(self.input_channels, self.structure[0], kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(self.structure[0])

        self.res0 = ResidualLayer(self.structure[0], self.structure[1])
        self.res1 = ResidualLayer(self.structure[1], self.structure[2])
        self.res2 = ResidualLayer(self.structure[2], self.structure[3])

        self.pooling = nn.MaxPool2d(kernel_size=4, stride=1, padding=0) 
        self.linear0 = nn.Linear(self.structure[3], self.structure[4])
        self.linear1 = nn.Linear(self.structure[4], self.out_classes)

        self.layers = [self.conv0, self.res0, self.res1, self.res2, self.linear0, self.linear1]
        self.representation = nn.Sequential(self.conv0, self.bn0, nn.ReLU(), self.res0, self.res1, self.res2, self.pooling)
        self.classifier = nn.Sequential(self.linear0, nn.ReLU(), self.linear1, nn.Softmax(dim=1))

    def set_size(self, size):
        self.size = size
        self.structure = self.structures[self.size]
        self.__set_structure()
    
    def set_structure(self, structure:list=None):
        if structure is not None:
            self.size = None
            self.structure = structure
            self.__set_structure()
       
    def forward(self, x):
        x = self.representation(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_macs(self, input_shape):
        return profile(self, inputs=(torch.empty(1, *input_shape),), verbose=False)[0]

    def get_weight_size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualLayer(nn.Module):
    def __init__(self, input_channels, out_channels) -> None:
        super(ResidualLayer, self).__init__()

        self.input_channels = input_channels
        self.out_channels = out_channels
        
        self.mainpath = nn.Sequential(
            nn.Conv2d(input_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(input_channels, out_channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        out = self.mainpath(x)
        residual = self.shortcut(x)
        out += residual
        out = F.relu(out)
        return out
