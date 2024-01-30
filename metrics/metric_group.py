from .accuracy_metric import AccuracyMetric
from .ece_metric import ECEMetric
from .sce_metric import SCEMetric


class MetricGroup(object):
    def __init__(self,
                 accuracyD=True,
                 accuracyI=True,
                 accuracyC=True,
                 ECED=False,
                 ECEI=False,
                 SCED=False,
                 SCEI=False,
                 q=1,
                 binary=False,
                 num_bins=10,
                 num_classes=19,
                 ignore_index=None):
        self.accuracy_metric = AccuracyMetric(accuracyD, accuracyI, accuracyC, q, binary, num_classes, ignore_index)
        self.ece_metric = ECEMetric(ECED, ECEI, num_bins, ignore_index)
        self.sce_metric = SCEMetric(SCED, SCEI, num_bins, num_classes, ignore_index)


    def add(self, prob, label, image_file=None):
        pred = prob.argmax(1)
        self.accuracy_metric.add(pred, label, image_file)
        self.ece_metric.add(prob, label)
        self.sce_metric.add(prob, label)


    def value(self):
        results = {}
        results.update(self.accuracy_metric.value())
        results.update(self.ece_metric.value())
        results.update(self.sce_metric.value())

        return results
