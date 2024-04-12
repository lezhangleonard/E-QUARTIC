import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

class DSConv(nn.Module):
    def __init__(self, inp, oup, stride=1, kernel_size=(3, 3), padding='same'):
        super(DSConv, self).__init__()
        self.depthwise = nn.Conv2d(inp, inp, kernel_size=kernel_size, stride=stride, padding=padding, groups=inp, bias=False)
        self.pointwise = nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(inp)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(oup)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class DSCNN(nn.Module):
    def __init__(self, input_channels, out_classes, size=1):
        super(DSCNN, self).__init__()
        self.out_classes = out_classes
        self.input_channels = input_channels

        # self.structures = {1: [64,64,64,64,64,64],
        #                    2: [32,36,48,48,48,48],
        #                    3: [24,32,32,36,44,36],
        #                    4: [20,24,28,30,38,32],
        #                    5: [18,20,24,26,32,24]
        #                   }

        self.structures = {1: [64,64,64,64,64,64],
                           4: [20,24,28,30,36,32]
                          }

        self.size = size
        self.structure = None
        self.set_size(size)
        self._initialize_weights()

    def _set_structure(self):
        self.initial_conv = nn.Conv2d(in_channels=self.input_channels, out_channels=self.structure[0], kernel_size=(10, 4), stride=(2, 2), bias=False)
        self.initial_bn = nn.BatchNorm2d(self.structure[0])
        self.initial_relu = nn.ReLU(inplace=True)
        self.initial_dropout = nn.Dropout(p=0.2)

        self.dsconv0 = DSConv(self.structure[0], self.structure[1])
        self.dsconv1 = DSConv(self.structure[1], self.structure[2])
        self.dsconv2 = DSConv(self.structure[2], self.structure[3])
        self.dsconv3 = DSConv(self.structure[3], self.structure[4])

        self.final_dropout = nn.Dropout(p=0.4)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.fc0 = nn.Linear(self.structure[4], self.structure[5])
        self.fc1 = nn.Linear(self.structure[5], self.out_classes)
        self.softmax = nn.Softmax(dim=1)

        self.layers = [self.initial_conv, self.dsconv0, self.dsconv1, self.dsconv2, self.dsconv3, self.fc0, self.fc1]
        self.representation = nn.Sequential(self.initial_conv, self.initial_bn, self.initial_relu, self.initial_dropout, self.dsconv0, self.dsconv1, self.dsconv2, self.dsconv3, self.final_dropout, self.avg_pool)
        self.classifier = nn.Sequential(self.fc0, nn.ReLU(), self.fc1, self.softmax)


    def set_size(self, size):
        self.size = size
        self.structure = self.structures[self.size]
        self._set_structure()

    def set_structure(self, structure:list=None):
        if structure is not None:
            self.size = None
            self.structure = structure
            self._set_structure()

    def forward(self, x):
        x = self.representation(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def get_macs(self, input_shape):
            return profile(self, inputs=(torch.empty(1, *input_shape),), verbose=False)[0]

    def get_weight_size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)