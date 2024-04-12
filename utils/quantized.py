import torch
import torch.nn as nn
import torch.nn.functional as F

class FixedPointQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale_factor):
        return (input * scale_factor).round()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None

def quantize(input, num_bits=16, num_frac_bits=10):
    scale_factor = 2**num_frac_bits
    return FixedPointQuantize.apply(input, scale_factor)

def dequantize(input, num_bits=16, num_frac_bits=10):
    scale_factor = 2**(-num_frac_bits)
    return input * scale_factor

class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(QuantizedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, input):
        # q_input = quantize(input)
        # q_weight = quantize(self.weight)
        # q_bias = quantize(self.bias)
        q_input = input
        q_weight = self.weight
        q_bias = self.bias
        output = F.linear(q_input, q_weight, q_bias)
        # output = dequantize(output)
        return output

class QuantizedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(QuantizedConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
    def forward(self, input):
        # q_input = quantize(input)
        # q_weight = quantize(self.conv.weight)
        # q_bias = self.conv.bias if self.conv.bias is None else quantize(self.conv.bias)
        q_input = input
        q_weight = self.conv.weight
        q_bias = self.conv.bias
        output = F.conv2d(q_input, q_weight, q_bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        # output = dequantize(output)
        return output

class QuantizedSoftmax(nn.Module):
    def __init__(self, dim=None):
        super(QuantizedSoftmax, self).__init__()
        self.dim = dim

    def forward(self, input):
        output = F.softmax(input, self.dim)
        return output
    
def copy_weights_and_freeze(from_model, to_model):
    for name, to_layer in to_model.named_children():
        if hasattr(from_model, name):
            from_layer = getattr(from_model, name)
            if isinstance(to_layer, QuantizedConv2d):
                with torch.no_grad():
                    to_layer.conv.weight.data = quantize(from_layer.weight.data)
                    if from_layer.bias is not None:
                        to_layer.conv.bias.data = quantize(from_layer.bias.data)
            elif isinstance(to_layer, QuantizedLinear):
                with torch.no_grad():
                    to_layer.weight.data = quantize(from_layer.weight.data)
                    if from_layer.bias is not None:
                        to_layer.bias.data = quantize(from_layer.bias.data)

            elif hasattr(to_layer, 'weight'):
                with torch.no_grad():
                    to_layer.weight.data = from_layer.weight.data.clone()
                    if hasattr(to_layer, 'bias') and to_layer.bias is not None:
                        to_layer.bias.data = from_layer.bias.data.clone()
                to_layer.weight.requires_grad = False
                if to_layer.bias is not None:
                    to_layer.bias.requires_grad = False
            

def get_trainable_params(model):
    params_to_update = [param for param in model.parameters() if param.requires_grad]
    return params_to_update
