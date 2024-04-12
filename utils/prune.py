import torch
import datasets
import copy
from torch.nn.utils import prune
from datasets.datasets import *
from models.baseline.resnet8 import ResidualLayer
from models.baseline.mobilenet import conv_bn, conv_1x1_bn, InvertedResidual, MobileNetV2
from models.baseline.mobilenetv1 import ConvBN, ConvDW, MobileNetV1
from models.baseline.dscnn import DSConv
from models.baseline.sqnet import Fire, SqueezeNet

def prune_model(model: torch.nn.Module, pruned_model: torch.nn.Module, structure: list, prune_structure: list) -> torch.nn.Module:
    previous_num_filters = None
    previous_filter_indices = None
    filter_indices = None
    for i in torch.arange(len(model.layers) - 1):
        layer = model.layers[i]
        if isinstance(layer, torch.nn.Conv2d):
            num_filters_to_keep = prune_structure[i]
            prune.ln_structured(layer, name='weight', amount=layer.out_channels - num_filters_to_keep, n=2, dim=0)
            filter_indices = torch.nonzero(layer.weight_mask.sum(dim=(1, 2, 3)), as_tuple=False).view(-1)
            prune.remove(layer, 'weight')
            if previous_filter_indices is not None:
                current_weight = layer.weight.data[filter_indices][:, previous_filter_indices].clone()
            else:
                current_weight = layer.weight.data[filter_indices].clone()
            pruned_model_layer = pruned_model.layers[i]
            pruned_model_layer.weight.data = current_weight
            pruned_model_layer.weight.requires_grad = True
            if pruned_model_layer.bias is not None:
                pruned_model_layer.bias.data = layer.bias.data[filter_indices].clone()
                pruned_model_layer.bias.requires_grad = True
            previous_filter_indices = filter_indices

        if isinstance(layer, ConvBN):
            sublayer = layer.conv[0]
            assert isinstance(sublayer, torch.nn.Conv2d)
            num_filters_to_keep = prune_structure[i]
            prune.ln_structured(sublayer, name='weight', amount=sublayer.out_channels - num_filters_to_keep, n=2, dim=0)
            filter_indices = torch.nonzero(sublayer.weight_mask.sum(dim=(1, 2, 3)), as_tuple=False).view(-1)
            prune.remove(sublayer, 'weight')
            if previous_filter_indices is not None:
                current_weight = sublayer.weight.data[filter_indices][:, previous_filter_indices].clone()
            else:
                current_weight = sublayer.weight.data[filter_indices].clone()
            pruned_model_layer = sublayer
            pruned_model_layer.weight.data = current_weight
            pruned_model_layer.weight.requires_grad = True
            if pruned_model_layer.bias is not None:
                pruned_model_layer.bias.data = sublayer.bias.data[filter_indices].clone()
                pruned_model_layer.bias.requires_grad = True
            previous_filter_indices = filter_indices

        if isinstance(layer, ConvDW):
            sublayer = layer.conv[0]
            assert isinstance(sublayer, torch.nn.Conv2d)
            num_filters_to_keep = prune_structure[i - 1] // sublayer.groups
            prune.ln_structured(sublayer, name='weight', amount=sublayer.out_channels - num_filters_to_keep, n=2, dim=0)
            filter_indices = torch.nonzero(sublayer.weight_mask.sum(dim=(1, 2, 3)), as_tuple=False).view(-1)
            prune.remove(sublayer, 'weight')
            if previous_filter_indices is not None:
                current_weight = sublayer.weight.data[filter_indices][:, previous_filter_indices].clone()
            else:
                current_weight = sublayer.weight.data[filter_indices].clone()
            pruned_model_layer = sublayer
            pruned_model_layer.weight.data = current_weight
            pruned_model_layer.weight.requires_grad = True
            if pruned_model_layer.bias is not None:
                pruned_model_layer.bias.data = sublayer.bias.data[filter_indices].clone()
                pruned_model_layer.bias.requires_grad = True
            previous_filter_indices = filter_indices

            sublayer = layer.conv[3]
            assert isinstance(sublayer, torch.nn.Conv2d)
            num_filters_to_keep = prune_structure[i] // sublayer.groups
            prune.ln_structured(sublayer, name='weight', amount=sublayer.out_channels - num_filters_to_keep, n=2, dim=0)
            filter_indices = torch.nonzero(sublayer.weight_mask.sum(dim=(1, 2, 3)), as_tuple=False).view(-1)
            prune.remove(sublayer, 'weight')
            if previous_filter_indices is not None:
                current_weight = sublayer.weight.data[filter_indices][:, previous_filter_indices].clone()
            else:
                current_weight = sublayer.weight.data[filter_indices].clone()
            pruned_model_layer = sublayer
            pruned_model_layer.weight.data = current_weight
            pruned_model_layer.weight.requires_grad = True
            if pruned_model_layer.bias is not None:
                pruned_model_layer.bias.data = sublayer.bias.data[filter_indices].clone()
                pruned_model_layer.bias.requires_grad = True
            previous_filter_indices = filter_indices

        if isinstance(layer, DSConv):
            sublayer = layer.depthwise
            assert isinstance(sublayer, torch.nn.Conv2d)
            num_filters_to_keep = prune_structure[i - 1] // sublayer.groups
            prune.ln_structured(sublayer, name='weight', amount=sublayer.out_channels - num_filters_to_keep, n=2, dim=0)
            filter_indices = torch.nonzero(sublayer.weight_mask.sum(dim=(1, 2, 3)), as_tuple=False).view(-1)
            prune.remove(sublayer, 'weight')
            if previous_filter_indices is not None:
                current_weight = sublayer.weight.data[filter_indices][:, previous_filter_indices].clone()
            else:
                current_weight = sublayer.weight.data[filter_indices].clone()
            pruned_model_layer = sublayer
            pruned_model_layer.weight.data = current_weight
            pruned_model_layer.weight.requires_grad = True
            if pruned_model_layer.bias is not None:
                pruned_model_layer.bias.data = sublayer.bias.data[filter_indices].clone()
                pruned_model_layer.bias.requires_grad = True
            previous_filter_indices = filter_indices

            sublayer = layer.pointwise
            assert isinstance(sublayer, torch.nn.Conv2d)
            num_filters_to_keep = prune_structure[i]
            prune.ln_structured(sublayer, name='weight', amount=sublayer.out_channels - num_filters_to_keep, n=2, dim=0)
            filter_indices = torch.nonzero(sublayer.weight_mask.sum(dim=(1, 2, 3)), as_tuple=False).view(-1)
            prune.remove(sublayer, 'weight')
            if previous_filter_indices is not None:
                current_weight = sublayer.weight.data[filter_indices][:, previous_filter_indices].clone()
            else:
                current_weight = sublayer.weight.data[filter_indices].clone()
            pruned_model_layer = sublayer
            pruned_model_layer.weight.data = current_weight
            pruned_model_layer.weight.requires_grad = True
            if pruned_model_layer.bias is not None:
                pruned_model_layer.bias.data = sublayer.bias.data[filter_indices].clone()
                pruned_model_layer.bias.requires_grad = True
            previous_filter_indices = filter_indices

        if isinstance(layer, Fire):
            sublayer = layer.squeeze
            assert isinstance(sublayer, torch.nn.Conv2d)
            num_filters_to_keep = layer.s1x1
            prune.ln_structured(sublayer, name='weight', amount=sublayer.out_channels - num_filters_to_keep, n=2, dim=0)
            filter_indices = torch.nonzero(sublayer.weight_mask.sum(dim=(1, 2, 3)), as_tuple=False).view(-1)
            prune.remove(sublayer, 'weight')
            if previous_filter_indices is not None:
                current_weight = sublayer.weight.data[filter_indices][:, previous_filter_indices].clone()
            else:
                current_weight = sublayer.weight.data[filter_indices].clone()
            pruned_model_layer = sublayer
            pruned_model_layer.weight.data = current_weight
            pruned_model_layer.weight.requires_grad = True
            if pruned_model_layer.bias is not None:
                pruned_model_layer.bias.data = sublayer.bias.data[filter_indices].clone()
                pruned_model_layer.bias.requires_grad = True
            previous_filter_indices = filter_indices

            sublayer = layer.expand1x1
            assert isinstance(sublayer, torch.nn.Conv2d)
            num_filters_to_keep = prune_structure[i] // 2
            prune.ln_structured(sublayer, name='weight', amount=sublayer.out_channels - num_filters_to_keep, n=2, dim=0)
            filter_indices = torch.nonzero(sublayer.weight_mask.sum(dim=(1, 2, 3)), as_tuple=False).view(-1)
            prune.remove(sublayer, 'weight')
            if previous_filter_indices is not None:
                current_weight = sublayer.weight.data[filter_indices][:, previous_filter_indices].clone()
            else:
                current_weight = sublayer.weight.data[filter_indices].clone()
            pruned_model_layer = sublayer
            pruned_model_layer.weight.data = current_weight
            pruned_model_layer.weight.requires_grad = True
            if pruned_model_layer.bias is not None:
                pruned_model_layer.bias.data = sublayer.bias.data[filter_indices].clone()
                pruned_model_layer.bias.requires_grad = True
            previous_filter_indices0 = filter_indices

            sublayer = layer.expand3x3
            assert isinstance(sublayer, torch.nn.Conv2d)
            num_filters_to_keep = prune_structure[i] // 2
            prune.ln_structured(sublayer, name='weight', amount=sublayer.out_channels - num_filters_to_keep, n=2, dim=0)
            filter_indices = torch.nonzero(sublayer.weight_mask.sum(dim=(1, 2, 3)), as_tuple=False).view(-1)
            prune.remove(sublayer, 'weight')
            if previous_filter_indices is not None:
                current_weight = sublayer.weight.data[filter_indices][:, previous_filter_indices].clone()
            else:
                current_weight = sublayer.weight.data[filter_indices].clone()
            pruned_model_layer = sublayer
            pruned_model_layer.weight.data = current_weight
            pruned_model_layer.weight.requires_grad = True
            if pruned_model_layer.bias is not None:
                pruned_model_layer.bias.data = sublayer.bias.data[filter_indices].clone()
                pruned_model_layer.bias.requires_grad = True
            previous_filter_indices1 = filter_indices

            previous_filter_indices = torch.cat((previous_filter_indices0, previous_filter_indices1), dim=0)

        if isinstance(layer, conv_bn) or isinstance(layer, conv_1x1_bn):
            sublayer = layer.conv[0]
            assert isinstance(sublayer, torch.nn.Conv2d)
            num_filters_to_keep = prune_structure[i]
            prune.ln_structured(sublayer, name='weight', amount=sublayer.out_channels - num_filters_to_keep, n=2, dim=0)
            filter_indices = torch.nonzero(sublayer.weight_mask.sum(dim=(1, 2, 3)), as_tuple=False).view(-1)
            prune.remove(sublayer, 'weight')
            if previous_filter_indices is not None:
                current_weight = sublayer.weight.data[filter_indices][:, previous_filter_indices].clone()
            else:
                current_weight = sublayer.weight.data[filter_indices].clone()
            pruned_model_layer = sublayer
            pruned_model_layer.weight.data = current_weight
            pruned_model_layer.weight.requires_grad = True
            pruned_model_layer.bias.data = sublayer.bias.data[filter_indices].clone()
            pruned_model_layer.bias.requires_grad = True
            previous_filter_indices = filter_indices

        if isinstance(layer, ResidualLayer):
            for sublayer in layer.mainpath:
                if isinstance(sublayer, torch.nn.Conv2d):
                    num_filters_to_keep = prune_structure[i]
                    prune.ln_structured(sublayer, name='weight', amount=sublayer.out_channels - num_filters_to_keep, n=2, dim=0)
                    filter_indices = torch.nonzero(sublayer.weight_mask.sum(dim=(1, 2, 3)), as_tuple=False).view(-1)
                    prune.remove(sublayer, 'weight')
                    if previous_filter_indices is not None:
                        current_weight = sublayer.weight.data[filter_indices][:, previous_filter_indices].clone()
                    else:
                        current_weight = sublayer.weight.data[filter_indices].clone()
                    pruned_model_layer = sublayer
                    pruned_model_layer.weight.data = current_weight
                    pruned_model_layer.weight.requires_grad = True
                    pruned_model_layer.bias.data = sublayer.bias.data[filter_indices].clone()
                    pruned_model_layer.bias.requires_grad = True

            for sublayer in layer.shortcut:
                if isinstance(sublayer, torch.nn.Conv2d):
                    num_filters_to_keep = prune_structure[i]
                    prune.ln_structured(sublayer, name='weight', amount=sublayer.out_channels - num_filters_to_keep, n=2, dim=0)
                    filter_indices = torch.nonzero(sublayer.weight_mask.sum(dim=(1, 2, 3)), as_tuple=False).view(-1)
                    prune.remove(sublayer, 'weight')
                    if previous_filter_indices is not None:
                        current_weight = sublayer.weight.data[filter_indices][:, previous_filter_indices].clone()
                    else:
                        current_weight = sublayer.weight.data[filter_indices].clone()
                    pruned_model_layer = sublayer
                    pruned_model_layer.weight.data = current_weight
                    pruned_model_layer.weight.requires_grad = True
                    pruned_model_layer.bias.data = sublayer.bias.data[filter_indices].clone()
                    pruned_model_layer.bias.requires_grad = True
            previous_filter_indices = filter_indices

        if isinstance(layer, InvertedResidual):
            for sublayer in layer.conv:
                if isinstance(layer, torch.nn.Conv2d):
                    num_filters_to_keep = prune_structure[i]
                    prune.ln_structured(sublayer, name='weight', amount=sublayer.out_channels - num_filters_to_keep, n=2, dim=0)
                    filter_indices = torch.nonzero(sublayer.weight_mask.sum(dim=(1, 2, 3)), as_tuple=False).view(-1)
                    prune.remove(sublayer, 'weight')
                    if previous_filter_indices is not None:
                        current_weight = sublayer.weight.data[filter_indices][:, previous_filter_indices].clone()
                    else:
                        current_weight = sublayer.weight.data[filter_indices].clone()
                    pruned_model_layer = sublayer
                    pruned_model_layer.weight.data = current_weight
                    pruned_model_layer.weight.requires_grad = True
                    pruned_model_layer.bias.data = sublayer.bias.data[filter_indices].clone()
                    pruned_model_layer.bias.requires_grad = True
            previous_filter_indices = filter_indices
            
        if isinstance(layer, torch.nn.Linear):
            # set uniform random weights
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0)
            # num_neurons_to_keep = prune_structure[i]
            # prune.ln_structured(layer, name='weight', amount=layer.out_features - num_neurons_to_keep, n=2, dim=0)
            # neuron_indices = torch.nonzero(layer.weight_mask.sum(dim=1), as_tuple=False).view(-1)
            # prune.remove(layer, 'weight')
            # if previous_filter_indices is not None and layer.in_features == structure[i-1]:
            #     current_weight = layer.weight.data[neuron_indices][:, previous_filter_indices].clone()
            # else:
            #     current_weight = layer.weight.data[neuron_indices].clone()
            # pruned_model_layer = pruned_model.layers[i]
            # pruned_model_layer.weight.data = current_weight
            # pruned_model_layer.weight.requires_grad = True
            # pruned_model_layer.bias.data = layer.bias.data[neuron_indices].clone()
            # pruned_model_layer.bias.requires_grad = True
    return pruned_model

def prune_random(model: torch.nn.Module, pruned_model: torch.nn.Module, structure: list, prune_structure: list) -> torch.nn.Module:
    previous_num_filters = None
    previous_filter_indices = None
    filter_indices = None
    for i in torch.arange(len(model.layers) - 1):
        layer = model.layers[i]
        if isinstance(layer, torch.nn.Conv2d):
            num_filters_to_keep = prune_structure[i]
            prune.random_structured(layer, name='weight', amount=layer.out_channels - num_filters_to_keep, dim=0)
            filter_indices = torch.nonzero(layer.weight_mask.sum(dim=(1, 2, 3)), as_tuple=False).view(-1)
            prune.remove(layer, 'weight')
            if previous_filter_indices is not None:
                current_weight = layer.weight.data[filter_indices][:, previous_filter_indices].clone()
            else:
                current_weight = layer.weight.data[filter_indices].clone()
            pruned_model_layer = pruned_model.layers[i]
            pruned_model_layer.weight.data = current_weight
            pruned_model_layer.weight.requires_grad = True
            if pruned_model_layer.bias is not None:
                pruned_model_layer.bias.data = layer.bias.data[filter_indices].clone()
                pruned_model_layer.bias.requires_grad = True
            previous_filter_indices = filter_indices

        if isinstance(layer, conv_bn) or isinstance(layer, conv_1x1_bn):
            sublayer = layer.conv[0]
            assert isinstance(sublayer, torch.nn.Conv2d)
            num_filters_to_keep = prune_structure[i]
            prune.random_structured(sublayer, name='weight', amount=sublayer.out_channels - num_filters_to_keep, dim=0)
            filter_indices = torch.nonzero(sublayer.weight_mask.sum(dim=(1, 2, 3)), as_tuple=False).view(-1)
            prune.remove(sublayer, 'weight')
            if previous_filter_indices is not None:
                current_weight = sublayer.weight.data[filter_indices][:, previous_filter_indices].clone()
            else:
                current_weight = sublayer.weight.data[filter_indices].clone()
            pruned_model_layer = sublayer
            pruned_model_layer.weight.data = current_weight
            pruned_model_layer.weight.requires_grad = True
            if pruned_model_layer.bias is not None:
                pruned_model_layer.bias.data = layer.bias.data[filter_indices].clone()
                pruned_model_layer.bias.requires_grad = True
            previous_filter_indices = filter_indices

        if isinstance(layer, ResidualLayer):
            for sublayer in layer.mainpath:
                if isinstance(sublayer, torch.nn.Conv2d):
                    num_filters_to_keep = prune_structure[i]
                    prune.random_structured(sublayer, name='weight', amount=sublayer.out_channels - num_filters_to_keep, dim=0)
                    filter_indices = torch.nonzero(sublayer.weight_mask.sum(dim=(1, 2, 3)), as_tuple=False).view(-1)
                    prune.remove(sublayer, 'weight')
                    if previous_filter_indices is not None:
                        current_weight = sublayer.weight.data[filter_indices][:, previous_filter_indices].clone()
                    else:
                        current_weight = sublayer.weight.data[filter_indices].clone()
                    pruned_model_layer = sublayer
                    pruned_model_layer.weight.data = current_weight
                    pruned_model_layer.weight.requires_grad = True
                    if pruned_model_layer.bias is not None:
                        pruned_model_layer.bias.data = layer.bias.data[filter_indices].clone()
                        pruned_model_layer.bias.requires_grad = True

            for sublayer in layer.shortcut:
                if isinstance(sublayer, torch.nn.Conv2d):
                    num_filters_to_keep = prune_structure[i]
                    prune.random_structured(sublayer, name='weight', amount=sublayer.out_channels - num_filters_to_keep, dim=0)
                    filter_indices = torch.nonzero(sublayer.weight_mask.sum(dim=(1, 2, 3)), as_tuple=False).view(-1)
                    prune.remove(sublayer, 'weight')
                    if previous_filter_indices is not None:
                        current_weight = sublayer.weight.data[filter_indices][:, previous_filter_indices].clone()
                    else:
                        current_weight = sublayer.weight.data[filter_indices].clone()
                    pruned_model_layer = sublayer
                    pruned_model_layer.weight.data = current_weight
                    pruned_model_layer.weight.requires_grad = True
                    if pruned_model_layer.bias is not None:
                        pruned_model_layer.bias.data = layer.bias.data[filter_indices].clone()
                        pruned_model_layer.bias.requires_grad = True
            previous_filter_indices = filter_indices

        if isinstance(layer, InvertedResidual):
            for sublayer in layer.conv:
                if isinstance(layer, torch.nn.Conv2d):
                    num_filters_to_keep = prune_structure[i]
                    prune.random_structured(sublayer, name='weight', amount=sublayer.out_channels - num_filters_to_keep, dim=0)
                    filter_indices = torch.nonzero(sublayer.weight_mask.sum(dim=(1, 2, 3)), as_tuple=False).view(-1)
                    prune.remove(sublayer, 'weight')
                    if previous_filter_indices is not None:
                        current_weight = sublayer.weight.data[filter_indices][:, previous_filter_indices].clone()
                    else:
                        current_weight = sublayer.weight.data[filter_indices].clone()
                    pruned_model_layer = sublayer
                    pruned_model_layer.weight.data = current_weight
                    pruned_model_layer.weight.requires_grad = True
                    if pruned_model_layer.bias is not None:
                        pruned_model_layer.bias.data = layer.bias.data[filter_indices].clone()
                        pruned_model_layer.bias.requires_grad = True
            previous_filter_indices = filter_indices
            
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_normal_(layer.weight)
            layer.bias.data.fill_(0)
            
            # num_neurons_to_keep = prune_structure[i]
            # prune.ln_structured(layer, name='weight', amount=layer.out_features - num_neurons_to_keep, n=2, dim=0)
            # neuron_indices = torch.nonzero(layer.weight_mask.sum(dim=1), as_tuple=False).view(-1)
            # prune.remove(layer, 'weight')
            # if previous_filter_indices is not None and layer.in_features == structure[i-1]:
            #     current_weight = layer.weight.data[neuron_indices][:, previous_filter_indices].clone()
            # else:
            #     current_weight = layer.weight.data[neuron_indices].clone()
            # pruned_model_layer = pruned_model.layers[i]
            # pruned_model_layer.weight.data = current_weight
            # pruned_model_layer.weight.requires_grad = True
            # pruned_model_layer.bias.data = layer.bias.data[neuron_indices].clone()
            # pruned_model_layer.bias.requires_grad = True
    return pruned_model
