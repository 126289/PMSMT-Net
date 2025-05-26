import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class BackboneClassifier(nn.Module):
    def __init__(self, backbone='resnet50', num_classes=3):
        super(BackboneClassifier, self).__init__()

        if backbone == 'resnet50':
            model = models.resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif backbone == 'vgg19':
            model = models.vgg19_bn(pretrained=True)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        elif backbone == 'efficientnet':
            model = models.efficientnet_b0(pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        else:
            raise ValueError("Unsupported backbone")

        self.model = model

    def forward(self, x):
        return self.model(x)


class EnsembleClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(EnsembleClassifier, self).__init__()
        self.net1 = BackboneClassifier('resnet50', num_classes)
        self.net2 = BackboneClassifier('vgg19', num_classes)
        self.net3 = BackboneClassifier('efficientnet', num_classes)

    def forward(self, x):
        out1 = self.net1(x)
        out2 = self.net2(x)
        out3 = self.net3(x)
        return out1, out2, out3

    def predict(self, x):
        out1, out2, out3 = self.forward(x)
        pred1 = torch.argmax(out1, dim=1)
        pred2 = torch.argmax(out2, dim=1)
        pred3 = torch.argmax(out3, dim=1)

        # Majority voting
        preds = torch.stack([pred1, pred2, pred3], dim=0)  # (3, B)
        final_pred = []
        for i in range(preds.size(1)):
            vals, counts = torch.unique(preds[:, i], return_counts=True)
            final_pred.append(vals[counts.argmax()])
        return torch.stack(final_pred)
