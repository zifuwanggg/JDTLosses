import torch

from .metric import Metric


class AccuracyMetric(Metric):
    def __init__(self, num_classes, ignore_index=None):
        super().__init__()
        self.correct = torch.ones(num_classes, dtype=torch.long)
        self.total = torch.ones(num_classes, dtype=torch.long)
        self.ignore_index = ignore_index


    def add(self, pred, label):
        keep_mask = label != self.ignore_index
        pred = pred[keep_mask]
        label = label[keep_mask]

        classes = label.unique().tolist()
        for i in classes:
            self.correct[i] += torch.sum(torch.logical_and(pred == i, label == i))
            self.total[i] += torch.sum(label == i)


    def value(self):
        Acc = 100 * torch.sum(self.correct) / torch.sum(self.total)
        mAcc = 100 * torch.mean(self.correct / self.total)

        return {"Acc": Acc, "mAcc": mAcc}
