import torch

from .metric import Metric, ConfusionMatrix


class mIoUDMetric(Metric):
    def __init__(self, num_classes, ignore_index=None):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = ConfusionMatrix(num_classes=num_classes)


    def add(self, pred, label):
        keep_mask = label != self.ignore_index
        pred = pred[keep_mask].reshape(-1)
        label = label[keep_mask].reshape(-1)

        if label.size(0) > 0:
            self.confusion_matrix.add(pred, label)


    def value(self):
        confusion_matrix = self.confusion_matrix.value()

        true_positives = torch.diag(confusion_matrix)
        false_positives = torch.sum(confusion_matrix, 0) - true_positives
        false_negatives = torch.sum(confusion_matrix, 1) - true_positives
        total = true_positives + false_positives + false_negatives

        if self.num_classes == 2:
            total[0] = 0

        if (total != 0).sum() == 0:
            return 1, 1

        true_positives = true_positives[total != 0]
        false_positives = false_positives[total != 0]
        false_negatives = false_negatives[total != 0]

        IoU = true_positives / \
            (true_positives + false_positives + false_negatives)
        Dice = 2 * true_positives / \
            (2 * true_positives + false_positives + false_negatives)

        mIoUD = 100 * torch.mean(IoU)
        mDiceD = 100 * torch.mean(Dice)

        return {"mIoUD": mIoUD, "mDiceD": mDiceD}
