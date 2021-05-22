import torch
import torch.nn as nn

class ModelD(nn.Module):

    def __init__(self, ebd, args):
        super(ModelD, self).__init__()

        self.args = args

        self.ebd = ebd

        self.d = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(500, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 2),
                )

    def forward(self, reverse_feature):

        logits = self.d(reverse_feature)  # [b, 500] -> [b, 2]

        return logits