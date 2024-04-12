import torch
import torch.nn as nn
from thop import profile
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, input_channels, out_classes, size=1):
        super(LeNet, self).__init__()
        self.structures = {1: [6,16,120,84],
                          2: [4,7,96,64],
                          3: [3,5,42,32],
                          4: [2,8,32,32],
                          5: [1,16,28,16],
                          6: [1,16,22,16],
                          7: [1,10,28,16],
                          8: [1,8,24,16],
                          9: [1,5,16,16],
                          10: [1,2,32,16]}
        self.size = size
        self.input_channels = input_channels
        self.out_classes = out_classes
        self.set_size(size)
    
    def _set_structure(self):
        self.conv0 = nn.Conv2d(self.input_channels, self.structure[0], kernel_size=5, stride=1, padding=0)
        self.maxpool0 = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(self.structure[0], self.structure[1], kernel_size=5, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(self.structure[1], self.structure[2], kernel_size=5, stride=1, padding=0)
        self.dropout = nn.Dropout(0.5)
        self.linear0 = nn.Linear(self.structure[2], self.structure[3])
        self.linear1 = nn.Linear(self.structure[3], self.out_classes)
        self.layers = [self.conv0, self.conv1, self.conv2, self.linear0, self.linear1]
        self.representation = nn.Sequential(self.conv0, nn.ReLU(), self.maxpool0, self.conv1, nn.ReLU(), self.maxpool1, self.conv2, nn.ReLU())
        self.classifier = nn.Sequential(self.linear0, nn.ReLU(), self.linear1, nn.Softmax(dim=1))
        
    def forward(self, x):
        x = self.representation(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def get_structure(self):
        return self.structure

    def set_size(self, size:int=1):
        self.size = size
        self.structure = self.structures[self.size]
        self._set_stucture()
    
    def set_structure(self, structure:list=None):
        if structure is not None:
            self.size = None
            self.structure = structure
            self._set_stucture()
    
    def get_macs(self, input_shape):
        return profile(self, inputs=(torch.empty(1, *input_shape),), verbose=False)[0]
    
    def get_weight_size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_representation(self):
        return self.representation
    
    def get_classifier(self):
        return self.classifier

class LeNet64x32(nn.Module):
    def __init__(self, input_channels, out_classes, size=1):
        super(LeNet64x32, self).__init__()
        self.structures = {1: [6,16,120,84],
                          2: [4,7,96,64],
                          3: [3,5,42,32],
                          4: [2,8,32,32],
                          5: [1,16,64,64],
                          6: [1,13,48,32],
                          7: [1,10,32,32],
                          8: [1,8,32,32],
                          9: [1,5,16,16],
                          10: [1,2,32,16]}
        self.size = size
        self.input_channels = input_channels
        self.out_classes = out_classes
        self.set_size(size)
    
    def _set_stucture(self):
        self.conv0 = nn.Conv2d(self.input_channels, self.structure[0], kernel_size=(5,5), stride=(1,1), padding=(0,1))
        self.maxpool0 = nn.MaxPool2d((4,2))
        self.conv1 = nn.Conv2d(self.structure[0], self.structure[1], kernel_size=5, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(self.structure[1], self.structure[2], kernel_size=5, stride=1, padding=0)
        self.linear0 = nn.Linear(self.structure[2], self.structure[3])
        self.linear1 = nn.Linear(self.structure[3], self.out_classes)
        self.layers = [self.conv0, self.conv1, self.conv2, self.linear0, self.linear1]
        self.representation = nn.Sequential(self.conv0, nn.ReLU(), self.maxpool0, self.conv1, nn.ReLU(), self.maxpool1, self.conv2, nn.ReLU())
        self.classifier = nn.Sequential(self.linear0, nn.ReLU(), self.linear1, nn.Softmax(dim=1))
        
    def forward(self, x):
        x = self.conv0(x)
        x = F.relu(x)
        x = self.maxpool0(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.linear0(x)
        x = F.relu(x)
        x = self.linear1(x)
        x = F.softmax(x, dim=1)
        return x
    
    def get_structure(self):
        return self.structure

    def set_size(self, size:int=1):
        self.size = size
        self.structure = self.structures[self.size]
        self._set_stucture()
    
    def set_structure(self, structure:list=None):
        if structure is not None:
            self.size = None
            self.structure = structure
            self._set_stucture()
    
    def get_macs(self, input_shape):
        return profile(self, inputs=(torch.empty(1, *input_shape),), verbose=False)[0]
    
    def get_weight_size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_representation(self):
        return self.representation
    
    def get_classifier(self):
        return self.classifier

class LeNet128x9(nn.Module):
    def __init__(self, input_channels, out_classes, size=1):
        super(LeNet128x9, self).__init__()
        self.structures = {1: [6,16,120,84],
                          2: [4,10,72,64],
                          3: [4,5,48,32],
                          4: [3,5,36,32],
                          5: [2,8,28,24],
                          6: [2,5,24,24],
                          7: [1,14,24,24],
                          8: [1,10,24,24],
                          9: [1,8,24,16],
                          10: [1,7,24,16]}
        self.size = size
        self.input_channels = input_channels
        self.out_classes = out_classes
        self.set_size(size)
    
    def _set_stucture(self):
        self.conv0 = nn.Conv2d(self.input_channels, self.structure[0], kernel_size=(8,3), stride=(2,1), padding=(3,1))
        self.maxpool0 = nn.MaxPool2d((2,1))
        self.conv1 = nn.Conv2d(self.structure[0], self.structure[1], kernel_size=(4,3), stride=(2,1), padding=(1,1))
        self.maxpool1 = nn.MaxPool2d((4,2))
        self.conv2 = nn.Conv2d(self.structure[1], self.structure[2], kernel_size=4, stride=1, padding=0)
        self.linear0 = nn.Linear(self.structure[2], self.structure[3])
        self.linear1 = nn.Linear(self.structure[3], self.out_classes)
        self.layers = [self.conv0, self.conv1, self.conv2, self.linear0, self.linear1]
        self.representation = nn.Sequential(self.conv0, nn.ReLU(), self.maxpool0, self.conv1, nn.ReLU(), self.maxpool1, self.conv2, nn.ReLU())
        self.classifier = nn.Sequential(self.linear0, nn.ReLU(), self.linear1, nn.Softmax(dim=1))
        
    def forward(self, x):
        x = self.representation(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def get_structure(self):
        return self.structure

    def set_size(self, size:int=1):
        self.size = size
        self.structure = self.structures[self.size]
        self._set_stucture()
    
    def set_structure(self, structure:list=None):
        if structure is not None:
            self.size = None
            self.structure = structure
            self._set_stucture()
    
    def get_macs(self, input_shape):
        return profile(self, inputs=(torch.empty(1, *input_shape),), verbose=False)[0]
    
    def get_weight_size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_representation(self):
        return self.representation
    
    def get_classifier(self):
        return self.classifier