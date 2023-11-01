import torch

from .metric import Metric, get_ece


class ECEDMetric(Metric):
    def __init__(self, num_bins, ignore_index=None):
        super().__init__()
        self.bins = torch.linspace(0., 1. - 1. / num_bins, num_bins)
        self.bin_acc = torch.zeros(num_bins)
        self.bin_conf = torch.zeros(num_bins)
        self.bin_total = torch.zeros(num_bins)
        self.ignore_index = ignore_index


    def add(self, prob, label):
        y_pred = prob.argmax(1)
        y_true = label == y_pred
        y_conf = prob.max(1)[0]

        keep_mask = label != self.ignore_index
        y_true = y_true[keep_mask].reshape(-1)
        y_conf = y_conf[keep_mask].reshape(-1)

        bin_ids = torch.bucketize(y_conf, self.bins, right=True) - 1
        self.bin_total += \
            torch.bincount(bin_ids, minlength=len(self.bins))
        self.bin_acc += \
            torch.bincount(bin_ids, weights=y_true, minlength=len(self.bins))
        self.bin_conf += \
            torch.bincount(bin_ids, weights=y_conf, minlength=len(self.bins))


    def value(self):
        ECED = get_ece(self.bin_total,
                       self.bin_acc,
                       self.bin_conf)

        return {"ECED": ECED}
