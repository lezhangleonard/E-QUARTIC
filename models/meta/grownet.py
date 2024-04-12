import torch
from torch import nn

class GrowNet(nn.Module):
    def __init__(self):
        super(GrowNet, self).__init__()
        self.learners = nn.ModuleList()

    def forward(self, x):
        if len(self.learners) == 0:
            return None
        outputs = []
        last_feature = None
        for learner in self.learners:
            outputs.append(learner([x, last_feature]))
            last_feature = learner.get_feature([x, last_feature])
        outputs = torch.stack(outputs, dim=0)
        outputs = outputs.mean(dim=0)
        return outputs
    
    def get_feature(self, x):
        if len(self.learners) == 0:
            return None
        features = []
        last_feature = None
        for learner in self.learners:
            features.append(learner.get_feature([x, last_feature]))
            last_feature = features[-1]
        return features

    def add_learner(self, learner):
        self.learners.append(learner.requires_grad_(False))


class WeakLearner(nn.Module):
    def __init__(self, base_model, first=False):
        super(WeakLearner, self).__init__()
        self.base_model = base_model
        self.representation = base_model.representation
        if first:
            self.feature = base_model.classifier[:-1]
        else:
            self.feature = nn.Sequential(
                nn.Linear(base_model.classifier[0].in_features + base_model.classifier[-2].out_features, base_model.classifier[0].out_features),
                *base_model.classifier[1:-1]
            )
        self.classifier = base_model.classifier[-1]
    
    def forward(self, x):
        inp, res = x[0], x[1]
        out = self.representation(inp)
        out = out.view(out.size(0), -1)
        if res is not None:
            out = torch.cat((out, res), dim=1)
        out = self.feature(out)
        out = self.classifier(out)
        return out
    
    def get_feature(self, x):
        inp, res = x[0], x[1]
        out = self.representation(inp)
        out = out.view(out.size(0), -1)
        if res is not None:
            out = torch.cat((out, res), dim=1)
        out = self.feature(out)
        return out