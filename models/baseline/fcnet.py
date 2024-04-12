import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from utils.quantized import QuantizedConv2d, QuantizedLinear, QuantizedSoftmax

class FCNN(nn.Module):
    def __init__(self, input_channels, out_classes, size=1):
        super(FCNN, self).__init__()
        self.input_channels = input_channels
        self.out_classes = out_classes
        self.structures = {1:[8,16,32,32,64,64,32],
                           2:[6,12,22,22,44,44,24],
                           3:[5,10,18,18,36,36,20],
                           4:[4,8,16,16,32,32,16]}

        self.size = size
        self.input_channels = input_channels
        self.out_classes = out_classes
        self.set_size(size)

    def _set_structure(self):
        self.conv1 = nn.Conv2d(self.input_channels, self.structure[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.structure[0], self.structure[1], kernel_size=3, padding=1)
        self.maxpool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn0 = nn.BatchNorm2d(self.structure[1])

        self.conv3 = nn.Conv2d(self.structure[1], self.structure[2], kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(self.structure[2], self.structure[3], kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(self.structure[3])

        self.conv5 = nn.Conv2d(self.structure[3], self.structure[4], kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(self.structure[4], self.structure[5], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.structure[5])

        self.dropout = nn.Dropout(0.2)

        self.linear0 = nn.Linear(self.structure[5] * 7 * 7, self.structure[6])
        self.linear1 = nn.Linear(self.structure[6], self.out_classes)

        self.layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.linear0, self.linear1]

        self.features = nn.Sequential(self.conv1, nn.ReLU(), self.conv2, nn.ReLU(), self.maxpool0, self.bn0, self.conv3, nn.ReLU(), self.conv4, nn.ReLU(), self.maxpool1, self.bn1, self.conv5, nn.ReLU(), self.conv6, nn.ReLU(), self.bn2)
        self.classifier = nn.Sequential(self.linear0, nn.ReLU(), self.dropout, self.linear1, nn.Softmax(dim=1))

    def set_size(self, size:int=1):
        self.size = size
        self.structure = self.structures[self.size]
        self._set_structure()

    def set_structure(self, structure:list=None):
        if structure is not None:
            self.size = None
            self.structure = structure
            self._set_structure()
    
    def get_macs(self, input_shape):
        return profile(self, inputs=(torch.empty(1, *input_shape),), verbose=False)[0]
    
    def get_weight_size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class QuantizedDCNN(nn.Module):
    def __init__(self, input_channels, out_classes, size=1):
        super(QuantizedDCNN, self).__init__()
        self.input_channels = input_channels
        self.out_classes = out_classes
        self.structures = {1: [8, 16, 32, 32, 64, 64, 32],
                           2: [6, 12, 22, 22, 44, 44, 24],
                           3: [5, 10, 18, 18, 36, 36, 20],
                           4: [4, 8, 16, 16, 32, 32, 16]}
        self.size = size
        self.set_size(size)

    def _set_structure(self):
        self.conv1 = nn.Conv2d(self.input_channels, self.structure[0], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.structure[0], self.structure[1], kernel_size=3, padding=1)
        self.maxpool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn0 = nn.BatchNorm2d(self.structure[1])

        self.conv3 = nn.Conv2d(self.structure[1], self.structure[2], kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(self.structure[2], self.structure[3], kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(self.structure[3])

        self.conv5 = nn.Conv2d(self.structure[3], self.structure[4], kernel_size=3, padding=1)
        self.conv6 = QuantizedConv2d(self.structure[4], self.structure[5], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(self.structure[5])

        self.dropout = nn.Dropout(0.2)

        self.linear0 = QuantizedLinear(self.structure[5] * 7 * 7, self.structure[6])
        self.linear1 = QuantizedLinear(self.structure[6], self.out_classes)

        self.features = nn.Sequential(self.conv1, nn.ReLU(), self.conv2, nn.ReLU(), self.maxpool0, self.bn0, 
                                      self.conv3, nn.ReLU(), self.conv4, nn.ReLU(), self.maxpool1, self.bn1, 
                                      self.conv5, nn.ReLU(), self.conv6, nn.ReLU(), self.bn2)
        self.classifier = nn.Sequential(self.linear0, nn.ReLU(), self.dropout, self.linear1, QuantizedSoftmax(dim=1))

    def set_size(self, size: int = 1):
        self.size = size
        self.structure = self.structures[self.size]
        self._set_structure()

    def set_structure(self, structure: list = None):
        if structure is not None:
            self.size = None
            self.structure = structure
            self._set_structure()

    def get_macs(self, input_shape):
        return profile(self, inputs=(torch.empty(1, *input_shape),), verbose=False)[0]

    def get_weight_size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x