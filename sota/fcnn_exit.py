import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

class FCNN(nn.Module):
    def __init__(self, input_channels, out_classes, size=1):
        super(FCNN, self).__init__()
        self.input_channels = input_channels
        self.out_classes = out_classes
        self.structures = {1:[8,16,32,32,64,64,32]}

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
        self.exit0 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Flatten(),
            nn.Linear(288, 10),
            nn.Softmax(dim=1)
        )
        self.conv4 = nn.Conv2d(self.structure[2], self.structure[3], kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(self.structure[3])

        self.exit1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(288, 10),
            nn.Softmax(dim=1)
        )

        self.conv5 = nn.Conv2d(self.structure[3], self.structure[4], kernel_size=3, padding=1)

        self.exit2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(576, 10),
            nn.Softmax(dim=1)
        )

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
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool0(x)
        x = self.bn0(x)

        x = self.conv3(x)
        x = F.relu(x)
        x0 = self.exit0(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.bn1(x)
        x1 = self.exit1(x)

        x = self.conv5(x)
        x = F.relu(x)
        x2 = self.exit2(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = x.view(x.size(0), -1)
        x = self.linear0(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear1(x)
        x = F.softmax(x, dim=1)
        return [x0, x1, x2, x]


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
