import torch
import torch.nn as nn

class MultiInputClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(MultiInputClassifier, self).__init__()

        def feature_branch():
            return nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)
            )

        self.seg_branch = feature_branch()
        self.edge_branch = feature_branch()
        self.morph_branch = feature_branch()

        self.classifier = nn.Sequential(
            nn.Linear(32 * 3, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, seg, edge, morph):
        f1 = self.seg_branch(seg)
        f2 = self.edge_branch(edge)
        f3 = self.morph_branch(morph)
        f = torch.cat([f1, f2, f3], dim=1).view(seg.size(0), -1)
        return self.classifier(f)
