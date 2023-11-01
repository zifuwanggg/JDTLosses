import torch


class Metric(object):
    def add(self):
        pass

    def value(self):
        pass


class ConfusionMatrix(Metric):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)


    def add(self, pred, label):
        assert pred.shape == label.shape
        assert (pred.max() < self.num_classes) and (pred.min() >= 0)
        assert (label.max() < self.num_classes) and (label.min() >= 0)

        x = pred + self.num_classes * label
        bincount = torch.bincount(x.long(), minlength=self.num_classes ** 2)
        self.confusion_matrix += bincount.reshape((self.num_classes, self.num_classes))


    def value(self):
        return self.confusion_matrix


def get_ece(bin_total, bin_acc, bin_conf):
    nonzero = bin_total != 0
    weights = bin_total[nonzero] / torch.sum(bin_total[nonzero])
    prob_acc = (bin_acc[nonzero] / bin_total[nonzero])
    prob_conf = (bin_conf[nonzero] / bin_total[nonzero])
    diff = torch.abs(prob_acc - prob_conf)
    ece = 100 * torch.sum(weights * diff)

    return ece
