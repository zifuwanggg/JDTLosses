"""
Jaccard Metric Losses: Optimizing the Jaccard Index with Soft Labels <https://arxiv.org/abs/2302.05666>
Dice Semimetric Losses: Optimizing the Dice Score with Soft Labels <https://arxiv.org/abs/2303.16296>
Revisiting Evaluation Metrics for Semantic Segmentation: Optimization and Evaluation of Fine-grained Intersection over Union <https://arxiv.org/abs/2310.19252>
"""

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class JDTLoss(_Loss):
    def __init__(self,
                 mIoUD=1.0,
                 mIoUI=0.0,
                 mIoUC=0.0,
                 alpha=1.0,
                 beta=1.0,
                 gamma=1.0,
                 smooth=1e-3,
                 threshold=0.01,
                 norm=1,
                 log_loss=False,
                 ignore_index=None,
                 class_weights=None,
                 active_classes_mode_hard="PRESENT",
                 active_classes_mode_soft="ALL"):
        """
        Arguments:
            mIoUD (float): The weight of the loss to optimize mIoUD.
            mIoUI (float): The weight of the loss to optimize mIoUI.
            mIoUC (float): The weight of the loss to optimize mIoUC.
            alpha (float): The coefficient of false positives in the Tversky loss.
            beta (float): The coefficient of false negatives in the Tversky loss.
            gamma (float): When `gamma` > 1, the loss focuses more on
                less accurate predictions that have been misclassified.
            smooth (float): A floating number to avoid `NaN` error.
            threshold (float): The threshold to select active classes.
            norm (int): The norm to compute the cardinality.
            log_loss (bool): Compute the log loss or not.
            ignore_index (int | None): The class index to be ignored.
            class_weights (list[float] | None): The weight of each class.
                If it is `list[float]`, its size should be equal to the number of classes.
            active_classes_mode_hard (str): The mode to compute
                active classes when training with hard labels.
            active_classes_mode_soft (str): The mode to compute
                active classes when training with hard labels.

        Comments:
            Jaccard: `alpha`  = 1.0, `beta`  = 1.0
            Dice:    `alpha`  = 0.5, `beta`  = 0.5
            Tversky: `alpha` >= 0.0, `beta` >= 0.0
        """
        super().__init__()

        assert mIoUD >= 0 and mIoUI >= 0 and mIoUC >= 0 and \
               alpha >= 0 and beta >= 0 and gamma >= 1 and \
               smooth >= 0 and threshold >= 0
        assert isinstance(norm, int) and norm > 0
        assert ignore_index == None or isinstance(ignore_index, int)
        assert class_weights == None or all((isinstance(w, float)) for w in class_weights)
        assert active_classes_mode_hard in ["ALL", "PRESENT"]
        assert active_classes_mode_soft in ["ALL", "PRESENT"]

        self.mIoUD = mIoUD
        self.mIoUI = mIoUI
        self.mIoUC = mIoUC
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.threshold = threshold
        self.norm = norm
        self.log_loss = log_loss
        self.ignore_index = ignore_index
        if class_weights == None:
            self.class_weights = class_weights
        else:
            self.class_weights = torch.tensor(class_weights)
        self.active_classes_mode_hard = active_classes_mode_hard
        self.active_classes_mode_soft = active_classes_mode_soft


    def forward(self, logits, label, keep_mask=None):
        """
        Arguments:
            logits (torch.Tensor): Its shape should be (B, C, D1, D2, ...).
            label (torch.Tensor):
                If it is hard label, its shape should be (B, D1, D2, ...).
                If it is soft label, its shape should be (B, C, D1, D2, ...).
            keep_mask (torch.Tensor | None):
                If it is `torch.Tensor`,
                    its shape should be (B, D1, D2, ...) and
                    its dtype should be `torch.bool`.
        """
        batch_size, num_classes = logits.shape[:2]
        hard_label = label.dtype == torch.long

        logits = logits.view(batch_size, num_classes, -1)
        prob = logits.log_softmax(dim=1).exp()

        if keep_mask != None:
            assert keep_mask.dtype == torch.bool
            keep_mask = keep_mask.view(batch_size, -1)
            keep_mask = keep_mask.unsqueeze(1).expand_as(prob)
        elif self.ignore_index != None and hard_label:
            keep_mask = label != self.ignore_index
            keep_mask = keep_mask.view(batch_size, -1)
            keep_mask = keep_mask.unsqueeze(1).expand_as(prob)

        if hard_label:
            label = torch.clamp(label, 0, num_classes - 1).view(batch_size, -1)
            label = F.one_hot(label, num_classes=num_classes).permute(0, 2, 1).float()
            active_classes_mode = self.active_classes_mode_hard
        else:
            label = label.view(batch_size, num_classes, -1)
            active_classes_mode = self.active_classes_mode_soft

        loss = self.forward_loss(prob, label, keep_mask, active_classes_mode)

        return loss


    def forward_loss(self, prob, label, keep_mask, active_classes_mode):
        if keep_mask != None:
            prob = prob * keep_mask
            label = label * keep_mask

        prob_card = torch.norm(prob, p=self.norm, dim=2)
        label_card = torch.norm(label, p=self.norm, dim=2)
        diff_card = torch.norm(prob - label, p=self.norm, dim=2)

        if self.norm > 1:
            prob_card = torch.pow(prob_card, exponent=self.norm)
            label_card = torch.pow(label_card, exponent=self.norm)
            diff_card = torch.pow(diff_card, exponent=self.norm)

        tp = (prob_card + label_card - diff_card) / 2
        fp = prob_card - tp
        fn = label_card - tp

        loss = 0
        batch_size, num_classes = prob.shape[:2]
        if self.mIoUD > 0:
            active_classes = self.compute_active_classes(label, active_classes_mode, num_classes, (0, 2))
            loss_mIoUD = self.forward_loss_mIoUD(tp, fp, fn, active_classes)
            loss += self.mIoUD * loss_mIoUD

        if self.mIoUI > 0 or self.mIoUC > 0:
            active_classes = self.compute_active_classes(label, active_classes_mode, (batch_size, num_classes), (2, ))
            loss_mIoUI, loss_mIoUC = self.forward_loss_mIoUIC(tp, fp, fn, active_classes)
            loss += self.mIoUI * loss_mIoUI + self.mIoUC * loss_mIoUC

        return loss


    def compute_active_classes(self, label, active_classes_mode, shape, dim):
        if active_classes_mode == "ALL":
            mask = torch.ones(shape, dtype=torch.bool)
        elif active_classes_mode == "PRESENT":
            mask = torch.amax(label, dim) > self.threshold

        active_classes = torch.zeros(shape, dtype=torch.bool, device=label.device)
        active_classes[mask] = 1

        return active_classes


    def forward_loss_mIoUD(self, tp, fp, fn, active_classes):
        if torch.sum(active_classes) < 1:
            return 0. * torch.sum(tp)

        tp = torch.sum(tp, dim=0)
        fp = torch.sum(fp, dim=0)
        fn = torch.sum(fn, dim=0)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        if self.log_loss:
            loss_mIoUD = -torch.log(tversky)
        else:
            loss_mIoUD = 1.0 - tversky

        if self.gamma > 1:
            loss_mIoUD **= self.gamma

        if self.class_weights != None:
            loss_mIoUD *= self.class_weights

        loss_mIoUD = loss_mIoUD[active_classes]
        loss_mIoUD = torch.mean(loss_mIoUD)

        return loss_mIoUD


    def forward_loss_mIoUIC(self, tp, fp, fn, active_classes):
        if torch.sum(active_classes) < 1:
            return 0. * torch.sum(tp), 0. * torch.sum(tp)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        if self.log_loss:
            loss_matrix = -torch.log(tversky)
        else:
            loss_matrix = 1.0 - tversky

        if self.gamma > 1:
            loss_matrix **= self.gamma

        if self.class_weights != None:
            class_weights = self.class_weights.unsqueeze(0).expand_as(loss_matrix)
            loss_matrix *= class_weights

        loss_matrix *= active_classes
        loss_mIoUI = self.reduce(loss_matrix, active_classes, 1)
        loss_mIoUC = self.reduce(loss_matrix, active_classes, 0)

        return loss_mIoUI, loss_mIoUC


    def reduce(self, loss_matrix, active_classes, dim):
        active_sum = torch.sum(active_classes, dim)
        active_dim = active_sum > 0
        loss = torch.sum(loss_matrix, dim)
        loss = loss[active_dim] / active_sum[active_dim]
        loss = torch.mean(loss)

        return loss
