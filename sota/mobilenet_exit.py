import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

class ConvBN(nn.Module):
    def __init__(self, inp, oup, stride):
        super(ConvBN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class ConvDW(nn.Module):
    def __init__(self, inp, oup, stride):
        super(ConvDW, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU6(inplace=True),
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class MobileNetV1(nn.Module):
    def __init__(self, input_channels, out_classes, size=1):
        super(MobileNetV1, self).__init__()
        self.num_classes = out_classes
        self.input_channels = input_channels

        # self.structures = {1: [8,16,32,32,64,64,128,128,128,128,128,128,256,256,256],
        #                    2: [6,12,24,24,48,48,64,64,64,72,96,96,128,128,128],
        #                    3: [6,8,16,24,32,48,48,48,48,64,64,64,64,64,64],
        #                    4: [5,8,16,16,24,32,32,32,48,48,48,64,64,64,64],
        #                    5: [4,8,16,16,16,24,24,32,32,32,48,48,48,48,48]
        #                   }

        self.structures = {1: [8,16,32,32,64,64,128,128,128,128,128,128,256,256,256],
                          }

        self.size = size
        self.structure = None
        self.set_size(size)
        
    def _set_structure(self):
        self.bn = ConvBN(self.input_channels, self.structure[0], 2)
        self.dw0 = ConvDW(self.structure[0], self.structure[1], 1)
        self.dw1 = ConvDW(self.structure[1], self.structure[2], 2)
        self.exit0 = nn.Sequential(
            nn.MaxPool2d(4, 4),
            nn.Flatten(),
            nn.Linear(1152, 2),
            nn.Softmax(dim=1)
        )
        self.dw2 = ConvDW(self.structure[2], self.structure[3], 1)
        self.dw3 = ConvDW(self.structure[3], self.structure[4], 2)
        self.dw4 = ConvDW(self.structure[4], self.structure[5], 1)
        self.exit1 = nn.Sequential(
            nn.MaxPool2d(4, 4),
            nn.Flatten(),
            nn.Linear(576, 2),
            nn.Softmax(dim=1)
        )
        self.dw5 = ConvDW(self.structure[5], self.structure[6], 2)
        self.dw6 = ConvDW(self.structure[6], self.structure[7], 1)
        self.dw7 = ConvDW(self.structure[7], self.structure[8], 1)
        self.exit2 = nn.Sequential(
            nn.MaxPool2d(4, 4),
            nn.Flatten(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )
        self.dw8 = ConvDW(self.structure[8], self.structure[9], 1)
        self.dw9 = ConvDW(self.structure[9], self.structure[10], 1)
        self.dw10 = ConvDW(self.structure[10], self.structure[11], 1)
        self.exit3 = nn.Sequential(
            nn.MaxPool2d(4, 4),
            nn.Flatten(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )
        self.dw11 = ConvDW(self.structure[11], self.structure[12], 2)
        self.dw12 = ConvDW(self.structure[12], self.structure[13], 1)
        self.pool = nn.AvgPool2d(3)
        self.fc0 = nn.Linear(self.structure[13], self.structure[14])
        self.fc1 = nn.Linear(self.structure[14], self.num_classes)

        self.layers = [self.bn, self.dw0, self.dw1, self.dw2, self.dw3, self.dw4, self.dw5, self.dw6, self.dw7, self.dw8, self.dw9, self.dw10, self.dw11, self.dw12, self.fc0, self.fc1]

        self.representation = nn.Sequential(self.bn, self.dw0, self.dw1, self.dw2, self.dw3, self.dw4, self.dw5, self.dw6, self.dw7, self.dw8, self.dw9, self.dw10, self.dw11, self.dw12, self.pool)
        self.classifier = nn.Sequential(self.fc0, nn.ReLU(), self.fc1, nn.Softmax(dim=1))

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
        x = self.bn(x)
        x = self.dw0(x)
        x = self.dw1(x)
        x0 = self.exit0(x)
        x = self.dw2(x)
        x = self.dw3(x)
        x1 = self.exit1(x)
        x = self.dw4(x)
        x = self.dw5(x)
        x = self.dw6(x)
        x2 = self.exit2(x)
        x = self.dw7(x)
        x = self.dw8(x)
        x = self.dw9(x)
        x3 = self.exit3(x)
        x = self.dw10(x)
        x = self.dw11(x)
        x = self.dw12(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc0(x)
        x = F.relu(x)
        x = self.fc1(x)
        x = F.softmax(x, dim=1)
        return [x0, x1, x2, x3, x]

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

    def freeze(self):
        for layer in self.layers:
            for param in layer.parameters():
                param.requires_grad = True
        for param in self.exit0.parameters():
            param.requires_grad = False
        for param in self.exit1.parameters():
            param.requires_grad = False
        for param in self.exit2.parameters():
            param.requires_grad = False
        for param in self.exit3.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for layer in self.layers:
            for param in layer.parameters():
                param.requires_grad = True
        for param in self.exit0.parameters():
            param.requires_grad = True
        for param in self.exit1.parameters():
            param.requires_grad = True
        for param in self.exit2.parameters():
            param.requires_grad = True
        for param in self.exit3.parameters():
            param.requires_grad = True

