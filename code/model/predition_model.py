import torch.nn as nn
from .bert import SBERT

class SBERTPrediction(nn.Module):
    """
    Pre-training task: predicting contaminated observations given an entire annual satellite time series
    """

    def __init__(self, sbert: SBERT, num_features):
        super().__init__()
        self.sbert = sbert
        self.mask_prediction = MaskedTimeSeriesModel(self.sbert.hidden, num_features)

    def forward(self, x, doy, mask):
        x = self.sbert(x, doy, mask)
        return self.mask_prediction(x)


class MaskedTimeSeriesModel(nn.Module):

    def __init__(self, hidden, num_features):
        super().__init__()
        self.linear = nn.Linear(hidden, num_features)

    def forward(self, x):
        return self.linear(x)
