import torch

from .metric import Metric, get_ece


class SCEIMetric(Metric):
    def __init__(self, num_bins, num_classes, ignore_index=None):
        super().__init__()
        self.sce = []
        self.num_bins = num_bins
        self.num_classes = num_classes
        self.ignore_index = ignore_index


    def add(self, prob, label):
        for i in range(prob.shape[0]):
            sce = 0
            bins = torch.linspace(0., 1. - 1. / self.num_bins, self.num_bins)
            bin_total = torch.zeros((self.num_classes, self.num_bins))
            bin_acc = torch.zeros((self.num_classes, self.num_bins))
            bin_conf = torch.zeros((self.num_classes, self.num_bins))

            prob_i = prob[i, ...]
            label_i = label[i, ...]

            y_pred = prob_i.argmax(0)
            correct = y_pred == label_i

            keep_mask = label_i != self.ignore_index
            label_i = torch.clamp(label_i, 0, self.num_classes - 1)

            if self.num_classes > 2:
                classes = label_i.unique().tolist()
            else:
                classes = [1]

            for j in classes:
                y_true = torch.logical_and(label_i == j, correct)
                y_conf = prob_i[j, ...]

                y_true = y_true[keep_mask].reshape(-1)
                y_conf = y_conf[keep_mask].reshape(-1)

                bin_ids = torch.bucketize(y_conf, bins, right=True) - 1
                bin_total[j] = torch.bincount(bin_ids, minlength=len(bins))
                bin_acc[j] = torch.bincount(bin_ids, weights=y_true, minlength=len(bins))
                bin_conf[j] = torch.bincount(bin_ids, weights=y_conf, minlength=len(bins))

                sce += get_ece(bin_total[j],
                               bin_acc[j],
                               bin_conf[j])

            self.sce.append(sce / len(classes))


    def value(self):
        # return permille
        SCEI = 10 * sum(self.sce) / len(self.sce)

        return {"SCEI": SCEI}
