import torch
import torch.nn as nn
from thop import profile

class conv_bn(nn.Module):
    def __init__(self, inp, oup, stride):
        super(conv_bn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )

class conv_1x1_bn(nn.Module):
    def __init__(self, inp, oup):
        super(conv_1x1_bn, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, input_channels, out_classes, size=1, input_size=32, width_mult=1.):
        super(MobileNetV2, self).__init__()
        self.block = InvertedResidual
        self.input_channels = input_channels
        self.out_classes = out_classes
        self.last_channel = 1280
        self.input_size = input_size
        self.width_mult = width_mult
        self.size = size
        self.input_size = input_size

        self.interverted_residual_settings = {
            1:[
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]],
            2:[
            [1, 8, 1, 1],
            [4, 16, 2, 2],
            [4, 24, 3, 2],
            [6, 48, 4, 2],
            [6, 64, 3, 1],
            [6, 128, 3, 2],
            [6, 320, 1, 1]],
            4:[
            [1, 6, 1, 1],
            [4, 10, 2, 2],
            [4, 12, 3, 2],
            [6, 24, 4, 2],
            [6, 40, 3, 1],
            [6, 84, 3, 2],
            [6, 256, 1, 1]],
            8:[
            [1, 3, 1, 1],
            [4, 6, 2, 2],
            [4, 6, 3, 2],
            [6, 12, 4, 2],
            [6, 20, 3, 1],
            [6, 56, 3, 2],
            [6, 216, 1, 1]]
        }

        self.set_size(self.size)

    def _set_structure(self):
        assert self.input_size % 32 == 0
        input_channels = int(self.input_channels * self.width_mult)
        self.last_channel = int(self.last_channel * self.width_mult) if self.width_mult > 1.0 else self.last_channel
        if self.input_size == 32:
            self.layers = [conv_bn(3, input_channels, 1).conv]
        else:
            self.layers = [conv_bn(3, input_channels, 2).conv]
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * self.width_mult)
            for i in range(n):
                if i == 0:
                    self.layers.append(self.block(input_channels, output_channel, s, expand_ratio=t))
                else:
                    self.layers.append(self.block(input_channels, output_channel, 1, expand_ratio=t))
                input_channels = output_channel
        self.layers.append(conv_1x1_bn(input_channels, self.last_channel).conv)
        self.representation = nn.Sequential(*self.layers)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, self.out_classes)
        )
        self._initialize_weights()

    def set_size(self, size):
        self.size = size
        self.interverted_residual_setting = self.interverted_residual_settings[self.size]
        self.structure = []
        self.expand_ratio = []
        for t, c, n, s in self.interverted_residual_setting:
            self.structure.append(c)
            self.expand_ratio.append(t)
        self._set_structure()
    
    def set_structure(self, structure:list=None):
        if structure is not None:
            self.size = None
            self.structure = structure
            for i in range(len(self.structure)):
                self.interverted_residual_setting[i][1] = self.structure[i]
            self._set_structure()

    def forward(self, x):
        x = self.representation(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        x = nn.functional.log_softmax(x, dim=1)
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