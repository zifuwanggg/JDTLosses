"""
Jaccard Metric Losses: Optimizing the Jaccard Index with Soft Labels
<https://arxiv.org/abs/2302.05666>
Dice Semimetric Losses: Optimizing the Dice Score with Soft Labels
<https://arxiv.org/abs/2303.16296>
Revisiting Evaluation Metrics for Semantic Segmentation: Optimization and Evaluation of Fine-grained Intersection over Union
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
                 smooth=1.0,
                 threshold=0.01,
                 active_classes_mode_hard="PRESENT",
                 active_classes_mode_soft="ALL",
                 class_weights=None,
                 ignore_index=None):
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
            active_classes_mode_hard (str): The mode to compute
                active classes when training with hard labels.
            active_classes_mode_soft (str): The mode to compute
                active classes when training with hard labels.
            class_weights (list[float] | None): The weight of each class.
                If it is `list[float]`, its size should be equal to the number of classes.
            ignore_index (int | None): The class index to be ignored.

        Comments:
            Jaccard: `alpha`  = 1.0, `beta`  = 1.0
            Dice:    `alpha`  = 0.5, `beta`  = 0.5
            Tversky: `alpha` >= 0.0, `beta` >= 0.0
        """
        super().__init__()

        assert mIoUD >= 0 and mIoUI >= 0 and mIoUC >= 0 and \
               alpha >= 0 and beta >= 0 and gamma >= 1 and \
               smooth >= 0 and threshold >= 0
        assert active_classes_mode_hard in \
               ["ALL", "PRESENT", "PROB", "LABEL", "BOTH"]
        assert active_classes_mode_soft in \
               ["ALL", "PRESENT", "PROB", "LABEL", "BOTH"]
        assert class_weights == None or \
               all((isinstance(w, float)) for w in class_weights)
        assert ignore_index == None or \
               isinstance(ignore_index, int)

        self.mIoUD = mIoUD
        self.mIoUI = mIoUI
        self.mIoUC = mIoUC
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.threshold = threshold
        self.active_classes_mode_hard = active_classes_mode_hard
        self.active_classes_mode_soft = active_classes_mode_soft
        if class_weights == None:
            self.class_weights = class_weights
        else:
            self.class_weights = torch.tensor(class_weights)
        self.ignore_index = ignore_index


    def forward(self,
                logits,
                label,
                keep_mask=None):
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
            keep_mask = keep_mask.unsqueeze(1).expand(batch_size, num_classes, -1)
        elif self.ignore_index != None and hard_label:
            keep_mask = label != self.ignore_index
            keep_mask = keep_mask.view(batch_size, -1)
            keep_mask = keep_mask.unsqueeze(1).expand(batch_size, num_classes, -1)

        if hard_label:
            label = label.view(batch_size, -1)
            label = F.one_hot(
                torch.clamp(label, 0, num_classes - 1), num_classes=num_classes)
            label = label.permute(0, 2, 1).float()
            active_classes_mode = self.active_classes_mode_hard
        else:
            label = label.view(batch_size, num_classes, -1)
            active_classes_mode = self.active_classes_mode_soft

        assert prob.shape == label.shape and \
               (keep_mask == None or prob.shape == keep_mask.shape)

        loss = self.forward_loss(prob,
                                 label,
                                 keep_mask,
                                 active_classes_mode)

        return loss


    def forward_loss(self,
                     prob,
                     label,
                     keep_mask,
                     active_classes_mode):
        if keep_mask != None:
            prob = prob * keep_mask
            label = label * keep_mask

        cardinality = torch.sum(prob + label, dim=2)
        difference = torch.sum(torch.abs(prob - label), dim=2)
        intersection = (cardinality - difference) / 2
        fp = torch.sum(prob, dim=2) - intersection
        fn = torch.sum(label, dim=2) - intersection

        loss = 0
        batch_size, num_classes = prob.shape[:2]
        if self.mIoUD > 0:
            active_classes = self.compute_active_classes(prob,
                                                         label,
                                                         active_classes_mode,
                                                         num_classes,
                                                         (0, 2))
            loss_mIoUD = self.forward_loss_mIoUD(intersection,
                                                 fp,
                                                 fn,
                                                 active_classes)
            loss += self.mIoUD * loss_mIoUD

        if self.mIoUI > 0 or self.mIoUC > 0:
            active_classes = self.compute_active_classes(prob,
                                                         label,
                                                         active_classes_mode,
                                                         (batch_size, num_classes),
                                                         (2, ))
            loss_mIoUI, loss_mIoUC = self.forward_loss_mIoUIC(intersection,
                                                              fp,
                                                              fn,
                                                              active_classes)
            loss += self.mIoUI * loss_mIoUI + self.mIoUC * loss_mIoUC

        return loss


    def compute_active_classes(self,
                               prob,
                               label,
                               active_classes_mode,
                               shape,
                               dim):
        if active_classes_mode == "ALL":
            mask = torch.ones(shape, dtype=torch.bool)
        elif active_classes_mode == "PRESENT":
            mask = torch.amax(label, dim) > 0.5
        elif active_classes_mode == "PROB":
            mask = torch.amax(prob, dim) > self.threshold
        elif active_classes_mode == "LABEL":
            mask = torch.amax(label, dim) > self.threshold
        elif active_classes_mode == "BOTH":
            mask = torch.amax(prob + label, dim) > self.threshold

        active_classes = torch.zeros(shape,
                                     dtype=torch.bool,
                                     device=prob.device)
        active_classes[mask] = 1

        return active_classes


    def forward_loss_mIoUD(self,
                           intersection,
                           fp,
                           fn,
                           active_classes):
        if torch.sum(active_classes) < 1:
            return 0. * torch.sum(intersection)

        intersection = torch.sum(intersection, dim=0)
        fp = torch.sum(fp, dim=0)
        fn = torch.sum(fn, dim=0)
        tversky = (intersection + self.smooth) / \
            (intersection + self.alpha * fp + self.beta * fn + self.smooth)

        loss_mIoUD = 1.0 - tversky
        if self.gamma > 1:
            loss_mIoUD **= self.gamma
        if self.class_weights != None:
            loss_mIoUD *= self.class_weights

        loss_mIoUD = loss_mIoUD[active_classes]
        loss_mIoUD = torch.mean(loss_mIoUD)

        return loss_mIoUD


    def forward_loss_mIoUIC(self,
                            intersection,
                            fp,
                            fn,
                            active_classes):
        if torch.sum(active_classes) < 1:
            return 0. * torch.sum(intersection), \
                   0. * torch.sum(intersection)

        tversky = (intersection + self.smooth) / \
            (intersection + self.alpha * fp + self.beta * fn + self.smooth)

        loss_matrix = 1.0 - tversky
        if self.gamma > 1:
            loss_matrix **= self.gamma
        if self.class_weights != None:
            class_weights = self.class_weights.unsqueeze(0).expand(loss_matrix.shape)
            loss_matrix *= class_weights

        loss_matrix *= active_classes
        loss_mIoUI = self.reduce(loss_matrix,
                                 active_classes,
                                 1)
        loss_mIoUC = self.reduce(loss_matrix,
                                 active_classes,
                                 0)

        return loss_mIoUI, loss_mIoUC


    def reduce(self,
               loss_matrix,
               active_classes,
               dim):
        loss = torch.sum(loss_matrix, dim)
        active_sum = torch.sum(active_classes, dim)
        active_dim = active_sum > 0
        loss = loss[active_dim] / active_sum[active_dim]
        loss = torch.mean(loss)

        return loss
