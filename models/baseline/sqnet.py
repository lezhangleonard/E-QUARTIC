import torch
import torch.nn as nn
import torch.nn.init as init
from thop import profile

class Fire(nn.Module):
    def __init__(self, in_channel, s1x1, e1x1, e3x3):
        super(Fire, self).__init__()
        self.s1x1 = s1x1
        self.e1x1 = e1x1
        self.e3x3 = e3x3
        self.squeeze = nn.Conv2d(in_channel, s1x1, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(s1x1, e1x1, kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(s1x1, e3x3, kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)

class SqueezeNet(nn.Module):
    def __init__(self, input_channels, out_classes, size=1):
        super(SqueezeNet, self).__init__()
        self.input_channels = input_channels
        self.out_classes = out_classes
        self.size = size
        self.structures = {1:[24, 32, 32, 64, 64, 96, 96, 128, 128, 128],
                           2:[16, 20, 20, 32, 32, 54, 54, 64, 64, 64],
                           3:[12, 14, 14, 22, 22, 32, 32, 44, 44, 44],
                           4:[8, 10, 10, 18, 18, 24, 24, 36, 36, 36],}
        self.structure = self.structures[self.size]
        self._set_structure()
        self.initialize()
        

    def _set_structure(self):
        self.conv0 = nn.Conv2d(self.input_channels, self.structure[0], kernel_size=3, padding=1)
        self.maxpool0 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.fire1 = Fire(self.structure[0], 8, self.structure[1]//2, self.structure[1]//2)
        self.fire2 = Fire(self.structure[1], 8, self.structure[2]//2, self.structure[2]//2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.fire3 = Fire(self.structure[2], 16, self.structure[3]//2, self.structure[3]//2)
        self.fire4 = Fire(self.structure[3], 16, self.structure[4]//2, self.structure[4]//2)
        self.fire5 = Fire(self.structure[4], 24, self.structure[5]//2, self.structure[5]//2)
        self.fire6 = Fire(self.structure[5], 24, self.structure[6]//2, self.structure[6]//2)
        self.fire7 = Fire(self.structure[6], 32, self.structure[7]//2, self.structure[7]//2)
        self.fire8 = Fire(self.structure[7], 32, self.structure[8]//2, self.structure[8]//2)

        self.dropout = nn.Dropout(0.2)
        self.conv1 = nn.Conv2d(self.structure[8], self.structure[9], kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc0 = nn.Linear(self.structure[9], self.out_classes)

        self.layers = [self.conv0, self.fire1, self.fire2, self.fire3, self.fire4, self.fire5, self.fire6, self.fire7, self.fire8, self.conv1, self.fc0]
        self.features = nn.Sequential(self.conv0, nn.ReLU(), self.maxpool0, self.fire1, self.fire2, self.maxpool1, self.fire3, self.fire4, self.fire5, self.fire6, self.fire7, self.fire8, self.dropout, self.conv1, nn.ReLU(), self.avgpool)
        self.classifier = nn.Sequential(self.fc0, nn.Softmax(dim=1))

    def set_size(self, size:int=1):
        self.size = size
        self.structure = self.structures[self.size]
        self._set_structure()
    
    def set_structure(self, structure:list=None):
        if structure is not None:
            self.size = None
            self.structure = structure
            self._set_structure()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def get_macs(self, input_shape):
        return profile(self, inputs=(torch.empty(1, *input_shape),), verbose=False)[0]
    
    def get_weight_size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
