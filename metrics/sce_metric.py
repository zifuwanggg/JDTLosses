import torch

from .metric import Metric, get_ece


class SCEMetric(Metric):
    def __init__(self,
                 SCED=True,
                 SCEI=True,
                 num_bins=10,
                 num_classes=19,
                 ignore_index=None):
        super().__init__()
        self.SCED = SCED
        self.SCEI = SCEI
        self.sce = []
        self.bins = torch.linspace(0., 1. - 1. / num_bins, num_bins)
        self.bin_acc = torch.zeros((num_classes, num_bins))
        self.bin_conf = torch.zeros((num_classes, num_bins))
        self.bin_total = torch.zeros((num_classes, num_bins))
        self.num_bins = num_bins
        self.num_classes = num_classes
        self.ignore_index = ignore_index


    def add(self, prob, label):
        if self.SCED:
            self.addD(prob, label)

        if self.SCEI:
            self.addI(prob, label)


    def value(self):
        results = {}

        if self.SCED:
            results.update(self.valueD())

        if self.SCEI:
            results.update(self.valueI())

        return results


    def addD(self, prob, label):
        y_pred = prob.argmax(1)
        correct = y_pred == label

        keep_mask = label != self.ignore_index

        for i in range(self.num_classes):
            y_true = torch.logical_and(label == i, correct)
            y_conf = prob[:, i, :, :]

            y_true = y_true[keep_mask].reshape(-1)
            y_conf = y_conf[keep_mask].reshape(-1)

            bin_ids = torch.bucketize(y_conf, self.bins, right=True) - 1
            self.bin_total[i] += torch.bincount(bin_ids, minlength=len(self.bins))
            self.bin_acc[i] += torch.bincount(bin_ids, weights=y_true, minlength=len(self.bins))
            self.bin_conf[i] += torch.bincount(bin_ids, weights=y_conf, minlength=len(self.bins))


    def addI(self, prob, label):
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


    def valueD(self):
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


    def valueI(self):
        # return permille
        SCEI = 10 * sum(self.sce) / len(self.sce)

        return {"SCEI": SCEI}
