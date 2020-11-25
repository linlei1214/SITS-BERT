import torch
import torch.nn as nn
from .bert import SBERT

class SBERTClassification(nn.Module):
    """
    Downstream task: Satellite Time Series Classification
    """

    def __init__(self, sbert: SBERT, num_classes):
        super().__init__()
        self.sbert = sbert
        self.classification = MulticlassClassification(self.sbert.hidden, num_classes)

    def forward(self, x, doy, mask):
        x = self.sbert(x, doy, mask)
        return self.classification(x, mask)


class MulticlassClassification(nn.Module):

    def __init__(self, hidden, num_classes):
        super().__init__()
        self.pooling = nn.MaxPool1d(64)
        self.linear = nn.Linear(hidden, num_classes)

    def forward(self, x, mask):
        x = self.pooling(x.permute(0, 2, 1)).squeeze()
        x = self.linear(x)
        return x