import torch
from torch import nn
from torch.nn.modules.loss import _Loss


# For hard label
class FocalLoss(_Loss):
    def __init__(self, gamma=0, ignore_index=None):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction="none",
                                           ignore_index=ignore_index)
        self.gamma = gamma
        self.ignore_index = ignore_index


    def forward(self, logits, label, keep_mask=None):
        if keep_mask == None and self.ignore_index != None:
            keep_mask = label != self.ignore_index

        logpt = -self.ce_loss(logits, label)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * logpt

        if keep_mask != None:
            loss = loss[keep_mask]

        loss = torch.mean(loss)

        return loss
