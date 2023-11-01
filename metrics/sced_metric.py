import torch

from .metric import Metric, get_ece


class SCEDMetric(Metric):
    def __init__(self, num_bins, num_classes, ignore_index=None):
        super().__init__()
        self.bins = torch.linspace(0., 1. - 1. / num_bins, num_bins)
        self.bin_acc = torch.zeros((num_classes, num_bins))
        self.bin_conf = torch.zeros((num_classes, num_bins))
        self.bin_total = torch.zeros((num_classes, num_bins))
        self.num_classes = num_classes
        self.ignore_index = ignore_index


    def add(self, prob, label):
        y_pred = prob.argmax(1)
        correct = y_pred == label

        keep_mask = label != self.ignore_index

        for i in range(self.num_classes):
            y_true = torch.logical_and(label == i, correct)
            y_conf = prob[:, i, :, :]

            y_true = y_true[keep_mask].reshape(-1)
            y_conf = y_conf[keep_mask].reshape(-1)

            bin_ids = torch.bucketize(y_conf, self.bins, right=True) - 1
            self.bin_total[i] += \
                torch.bincount(bin_ids, minlength=len(self.bins))
            self.bin_acc[i] += \
                torch.bincount(bin_ids, weights=y_true, minlength=len(self.bins))
            self.bin_conf[i] += \
                torch.bincount(bin_ids, weights=y_conf, minlength=len(self.bins))


    def value(self):
        if self.num_classes > 2:
            classes = range(self.num_classes)
        else:
            classes = range(1, 2)

        SCED = 0
        for i in classes:
            SCED += get_ece(self.bin_total[i],
                            self.bin_acc[i],
                            self.bin_conf[i])

        # return permille
        SCED *= (10 / len(classes))

        return {"SCED": SCED}
