import torch.nn as nn
import torch.nn.functional as F
from thop import profile
import torch

class ResNet8(nn.Module):
    def __init__(self, input_channels, out_classes, size=1):
        super(ResNet8, self).__init__()
        self.structures = {1: [16,32,64,128,64]}

        self.size = size
        self.input_channels = input_channels
        self.out_classes = out_classes
        self.set_size(size)

    def __set_structure(self):
        self.conv0 = nn.Conv2d(self.input_channels, self.structure[0], kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(self.structure[0])


        self.res0 = ResidualLayer(self.structure[0], self.structure[1])


        # self.exit0 = nn.Sequential(
        #     nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        #     nn.Conv2d(self.structure[1], 16, kernel_size=3, stride=2, padding=1),
        #     nn.Flatten(),
        #     nn.Linear(4*4*16, self.out_classes),
        #     nn.Softmax(dim=1)
        # )

        self.res1 = ResidualLayer(self.structure[1], self.structure[2])

        # self.exit1 = nn.Sequential(
        #     nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        #     nn.Conv2d(self.structure[2], 16, kernel_size=3, stride=2, padding=1),
        #     nn.Flatten(),
        #     nn.Linear(2*2*16, self.out_classes),
        #     nn.Softmax(dim=1)
        # )

        # self.exit2 = nn.Sequential(
        #     nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        #     nn.Conv2d(self.structure[2], 16, kernel_size=3, stride=2, padding=1),
        #     nn.Flatten(),
        #     nn.Linear(2*2*16, self.out_classes),
        #     nn.Softmax(dim=1)
        # )


        # self.exit4 = nn.Sequential(
        #     nn.MaxPool2d(kernel_size=4, stride=4, padding=0),
        #     nn.Flatten(),
        #     nn.Linear(8*8*self.structure[0], self.out_classes),
        #     nn.Softmax(dim=1)
        # )

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
        x = self.conv0(x)
        x = self.bn0(x)
        x = F.relu(x)

        # x4 = self.exit4(x)

        x = self.res0(x)
        # x0 = self.exit0(x)

        x1 = self.res1.mainpath[:3](x)
        x_main = self.res1.mainpath[3:](x1)
        x_short = self.res1.shortcut(x)
        x = x_main + x_short
        # x1 = self.exit1(x1)

        # x2 = self.exit2(x)

        x = self.res2(x)

        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x3 = self.linear0(x)
        x3 = F.relu(x3)
        x3 = self.linear1(x3)
        x3 = F.softmax(x3, dim=1)

        # return [x4, x0, x1, x2, x3]

    def get_macs(self, input_shape):
        return profile(self, inputs=(torch.empty(1, *input_shape),), verbose=False)[0]

    def get_weight_size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_main(self):
        for layer in self.layers:
            for param in layer.parameters():
                param.requires_grad = False
        for param in self.exit0.parameters():
            param.requires_grad = True
        for param in self.exit1.parameters():
            param.requires_grad = True
        for param in self.exit2.parameters():
            param.requires_grad = True

    def freeze_exit(self, step=None):
        for layer in self.layers:
            for param in layer.parameters():
                param.requires_grad = True
        if step is None:
            for param in self.exit0.parameters():
                param.requires_grad = False
            for param in self.exit1.parameters():
                param.requires_grad = False
            for param in self.exit2.parameters():
                param.requires_grad = False
            for param in self.exit4.parameters():
                param.requires_grad = False
            return
        elif step == 0:
            layers = [self.conv0, self.res0.mainpath[:3]]
        elif step == 1:
            layers = [self.conv0, self.res0, self.res1.mainpath[:3]]
        elif step == 2:
            layers = [self.conv0, self.res0, self.res1]
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False
        return

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
        for param in self.exit4.parameters():
            param.requires_grad = True
        
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
