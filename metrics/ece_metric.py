import torch

from .metric import Metric, get_ece


class ECEMetric(Metric):
    def __init__(self,
                 ECED=True,
                 ECEI=True,
                 num_bins=10,
                 ignore_index=None):
        super().__init__()
        self.ECED = ECED
        self.ECEI = ECEI
        self.ece = []
        self.bins = torch.linspace(0., 1. - 1. / num_bins, num_bins)
        self.bin_acc = torch.zeros(num_bins)
        self.bin_conf = torch.zeros(num_bins)
        self.bin_total = torch.zeros(num_bins)
        self.num_bins = num_bins
        self.ignore_index = ignore_index


    def add(self, prob, label):
        if self.ECED:
            self.addD(prob, label)

        if self.ECEI:
            self.addI(prob, label)


    def value(self):
        results = {}

        if self.ECED:
            results.update(self.valueD())

        if self.ECEI:
            results.update(self.valueI())

        return results


    def addD(self, prob, label):
        y_pred = prob.argmax(1)
        y_true = label == y_pred
        y_conf = prob.max(1)[0]

        keep_mask = label != self.ignore_index
        y_true = y_true[keep_mask].reshape(-1)
        y_conf = y_conf[keep_mask].reshape(-1)

        bin_ids = torch.bucketize(y_conf, self.bins, right=True) - 1
        self.bin_total += torch.bincount(bin_ids, minlength=len(self.bins))
        self.bin_acc += torch.bincount(bin_ids, weights=y_true, minlength=len(self.bins))
        self.bin_conf += torch.bincount(bin_ids, weights=y_conf, minlength=len(self.bins))


    def addI(self, prob, label):
        for i in range(prob.shape[0]):
            prob_i = prob[i, ...]
            label_i = label[i, ...]

            y_pred = prob_i.argmax(0)
            y_true = y_pred == label_i
            y_conf = prob_i.max(0)[0]

            keep_mask = label_i != self.ignore_index
            y_true = y_true[keep_mask].reshape(-1)
            y_conf = y_conf[keep_mask].reshape(-1)

            bins = torch.linspace(0., 1.- 1. / self.num_bins, self.num_bins)
            bin_ids = torch.bucketize(y_conf, bins, right=True) - 1
            bin_total = torch.bincount(bin_ids, minlength=len(bins))
            bin_acc = torch.bincount(bin_ids, weights=y_true, minlength=len(bins))
            bin_conf = torch.bincount(bin_ids, weights=y_conf, minlength=len(bins))

            ece = get_ece(bin_total, bin_acc, bin_conf)
            self.ece.append(ece)


    def valueD(self):
        ECED = get_ece(self.bin_total, self.bin_acc, self.bin_conf)

        return {"ECED": ECED}


    def valueI(self):
        ECEI = sum(self.ece) / len(self.ece)

        return {"ECEI": ECEI}

