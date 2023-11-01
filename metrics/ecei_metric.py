import torch

from .metric import Metric, get_ece


class ECEIMetric(Metric):
    def __init__(self, num_bins, ignore_index=None):
        super().__init__()
        self.ece = []
        self.num_bins = num_bins
        self.ignore_index = ignore_index


    def add(self, prob, label):
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

            ece = get_ece(bin_total,
                          bin_acc,
                          bin_conf)

            self.ece.append(ece)


    def value(self):
        ECEI = sum(self.ece) / len(self.ece)

        return {"ECEI": ECEI}
