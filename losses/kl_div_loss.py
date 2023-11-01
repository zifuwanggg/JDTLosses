import torch
from torch import nn
from torch.nn.modules.loss import _Loss


# For soft label
class KLDivLoss(_Loss):
    def __init__(self, T=1):
        super().__init__()
        self.kl_div_loss = nn.KLDivLoss(reduction="none")
        self.T = T


    def forward(self, logits, label, keep_mask=None):
        log_prob = (logits / self.T).log_softmax(dim=1)
        kl_div_loss = self.kl_div_loss(log_prob, label)
        loss = (self.T ** 2) * kl_div_loss.sum(dim=1)

        if keep_mask != None:
            loss = loss[keep_mask]

        loss = torch.mean(loss)

        return loss
