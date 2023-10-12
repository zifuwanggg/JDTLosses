"""
[1] Jaccard Metric Losses: Optimizing the Jaccard Index with Soft Labels
<https://arxiv.org/abs/2302.05666>.
[2] Dice Semimetric Losses: Optimizing the Dice Score with Soft Labels
<https://arxiv.org/abs/2303.16296>.
[3] Revisiting Evaluation Metrics for Semantic Segmentation: Optimization and Evaluation of Fine-grained Intersection over Union
"""

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class JDTLosses(_Loss):
    def __init__(self,
                 per_image=False,
                 log_loss=False,
                 from_logits=True,
                 T=1.0,
                 smooth=1.0,
                 alpha=1.0,
                 beta=1.0,
                 gamma=1.0,
                 threshold=0.01,
                 class_weight=None,
                 ignore_index=None):
        super().__init__()
        """
        Args:
            per_image (bool): Compute the loss per image or per batch.
            log_loss (bool): Compute the log loss or not.
            from_logits (bool): Inputs are logits or probabilities.
            T (float): Temperature to smooth predicted probabilities.
            smooth (float): A float number to avoid NaN error.
            alpha (float): The coefficient of false positives in the Tversky loss.
            beta (float): The coefficient of false negatives in the Tversky loss.
            gamma (float): When `gamma` > 1, the loss focuses more on less accurate predictions that have been misclassified.
            threshold (float): Threshold to select active classes.
            class_weight (torch.Tensor | list[float] | None): Weight of each class. If specified, its size should be equal to the number of classes.
            ignore_index (int | None): The class index to be ignored.

        Notes:
            Jaccard: `alpha`  = 1.0, `beta`  = 1.0
            Dice:    `alpha`  = 0.5, `beta`  = 0.5
            Tversky: `alpha` >= 0.0, `beta` >= 0.0
        """

        self.per_image = per_image
        self.log_loss = log_loss
        self.from_logits = from_logits
        self.T = T
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.threshold = threshold
        self.class_weight = class_weight
        self.ignore_index = ignore_index

        assert self.alpha >= 0 and self.beta >= 0


    def compute_active_classes(self, prob, label, classes):
        all_classes = torch.arange(prob.shape[0])

        if classes == "All":
            active_classes = all_classes
        elif classes == "Present":
            active_classes = torch.argmax(label, dim=0).unique()
        elif classes == "Prob":
            active_classes = all_classes[torch.amax(prob, dim=1) > self.threshold]
        elif classes == "Label":
            active_classes = all_classes[torch.amax(label, dim=1) > self.threshold]
        elif classes == "Both":
            active_classes = all_classes[torch.amax(prob + label, dim=1) > self.threshold]
        else:
            active_classes = torch.tensor(classes)

        return active_classes


    def compute_loss(self, prob, label):
        """
        Alternatives:
            union = (cardinality + difference) / 2
            jaccard = (intersection + self.smooth) / (union + self.smooth)
            dice = (2 * intersection + self.smooth) / (cardinality + self.smooth)
        """
        cardinality = torch.sum(prob + label, dim=1)
        difference = torch.sum(torch.abs(prob - label), dim=1)
        intersection = (cardinality - difference) / 2
        fp = torch.sum(prob, dim=1) - intersection
        fn = torch.sum(label, dim=1) - intersection
        tversky = (intersection + self.smooth) / \
            (intersection + self.alpha * fp + self.beta * fn + self.smooth)

        if self.log_loss:
            losses = -torch.log(tversky)
        else:
            losses = 1.0 - tversky

        if self.gamma > 1.0:
            losses **= (1.0 / self.gamma)

        if self.class_weight is not None:
            losses *= self.class_weight

        return losses


    def forward_per_image(self, prob, label, not_ignore, classes):
        num_classes, batch_size = prob.shape[:2]

        losses = ctn = 0

        for i in range(batch_size):
            not_ignore_i = not_ignore[:, i, :]
            prob_i = prob[:, i, :][not_ignore_i].reshape(num_classes, -1)
            label_i = label[:, i, :][not_ignore_i].reshape(num_classes, -1)

            if prob_i.size(1) < 1:
                continue

            active_classes = self.compute_active_classes(prob_i, label_i, classes)

            if active_classes.size(0) < 1:
                continue

            losses_i = self.compute_loss(prob_i, label_i)
            losses += losses_i[active_classes].mean()
            ctn += 1

        if ctn == 0:
            return 0. * prob.sum()

        return losses / ctn


    def forward_per_batch(self, prob, label, not_ignore, classes):
        """
        In distributed training, `forward_per_batch` computes the loss per GPU-batch instead of per whole batch.
        """

        num_classes = prob.shape[0]

        prob = prob.reshape(num_classes, -1)
        label = label.reshape(num_classes, -1)
        not_ignore = not_ignore.reshape(num_classes, -1)

        prob = prob[not_ignore].reshape(num_classes, -1)
        label = label[not_ignore].reshape(num_classes, -1)

        if prob.size(1) < 1:
            return 0. * prob.sum()

        active_classes = self.compute_active_classes(prob, label, classes)

        if active_classes.size(0) < 1:
            return 0. * prob.sum()

        losses = self.compute_loss(prob, label)

        return losses[active_classes].mean()


    def forward(self, logits, label, not_ignore=None, classes="Present"):
        """
        Args:
            logits (torch.Tensor): Logits or probabilities. Its shape should be (B, C, D1, D2, ...).
            label (torch.Tensor): When it is hard label, its shape should be (B, D1, D2, ...).
                                  When it is soft label, its shape should be (B, C, D1, D2, ...).
            not_ignore (torch.Tensor | None): (1) If `self.ignore_index` is `None`, it can be `None`.
                                              (2) If `self.ignore_index` is not `None` and `label` is hard label, it can be `None`.
                                              (3) In all other cases, its shape should be (B, D1, D2, ...) and its dtype should be `torch.bool`.
            classes (str | list[int]): When it is `str`, it is the mode to compute active classes.
                                       When it is `list[int]`, it is the list of class indices to compute the loss.
        """

        batch_size, num_classes = logits.shape[:2]
        hard_label = label.dtype == torch.long

        # (B, C, D1, D2, ...) -> (C, B, D1 * D2 * ...)
        logits = logits.view(batch_size, num_classes, -1).permute(1, 0, 2)

        if self.from_logits:
            prob = (logits / self.T).log_softmax(dim=0).exp()
        else:
            prob = logits

        if self.ignore_index is None and not_ignore is None:
            not_ignore = torch.ones_like(prob).to(torch.bool)
        elif self.ignore_index is not None and not_ignore is None and hard_label:
            not_ignore = (label != self.ignore_index).view(batch_size, -1).unsqueeze(0).expand(num_classes, batch_size, -1)
        else:
            not_ignore = not_ignore.view(batch_size, -1).unsqueeze(0).expand(num_classes, batch_size, -1)

        if hard_label:
            label = label.view(batch_size, -1)
            label = F.one_hot(torch.clamp(label, 0, num_classes - 1), num_classes=num_classes).permute(2, 0, 1)
        else:
            label = label.view(batch_size, num_classes, -1).permute(1, 0, 2)

        if self.class_weight is not None and not torch.is_tensor(self.class_weight):
            self.class_weight = prob.new_tensor(self.class_weight)

        assert prob.shape == label.shape == not_ignore.shape
        assert classes in ["All", "Present", "Prob", "Label", "Both"] or all((isinstance(c, int) and 0 <= c < num_classes) for c in classes)
        assert self.class_weight is None or self.class_weight.size(0) == num_classes

        if self.per_image:
            return self.forward_per_image(prob, label, not_ignore, classes)
        else:
            return self.forward_per_batch(prob, label, not_ignore, classes)
