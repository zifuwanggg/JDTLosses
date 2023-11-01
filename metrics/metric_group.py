import torch

from .accuracy_metric import AccuracyMetric
from .mioud_metric import mIoUDMetric
from .miouic_metric import mIoUICMetric
from .miouk_metric import mIoUKMetric
from .eced_metric import ECEDMetric
from .ecei_metric import ECEIMetric
from .sced_metric import SCEDMetric
from .scei_metric import SCEIMetric


class MetricGroup(object):
    def __init__(self,
                 Acc=False,
                 mIoUD=True,
                 mIoUIC=True,
                 mIoUK=False,
                 ECED=False,
                 ECEI=False,
                 SCED=False,
                 SCEI=False,
                 annos=None,
                 anno_map=None,
                 num_bins=10,
                 num_classes=19,
                 ignore_index=None,
                 reduce_panoptic_zero_label=False,
                 threshold=0,
                 q=5,
                 q_bar=True):
        if Acc:
            self.accuracy_metric = AccuracyMetric(num_classes, ignore_index)
        else:
            self.accuracy_metric = None

        if mIoUD:
            self.mioud_metric = mIoUDMetric(num_classes, ignore_index)
        else:
            self.mioud_metric = None

        if mIoUIC:
            self.miouic_metric = mIoUICMetric(num_classes, ignore_index)
        else:
            self.miouic_metric = None

        if mIoUK:
            self.miouk_metric = mIoUKMetric(annos,
                                            anno_map,
                                            num_classes,
                                            ignore_index,
                                            reduce_panoptic_zero_label,
                                            threshold)
        else:
            self.miouk_metric = None

        if ECED:
            self.eced_metric = ECEDMetric(num_bins, ignore_index)
        else:
            self.eced_metric = None

        if ECEI:
            self.ecei_metric = ECEIMetric(num_bins, ignore_index)
        else:
            self.ecei_metric = None

        if SCED:
            self.sced_metric = SCEDMetric(num_bins, num_classes, ignore_index)
        else:
            self.sced_metric = None

        if SCEI:
            self.scei_metric = SCEIMetric(num_bins, num_classes, ignore_index)
        else:
            self.scei_metric = None

        self.q = q
        self.q_bar = q_bar


    def add(self,
            prob,
            label,
            panoptic=None,
            image_file=None,
            panoptic_file=None,
            level=None):
        if level == None:
            pred = prob.argmax(1)
        else:
            prob_foreground = prob[:, 1, :, :]
            pred = torch.where(prob_foreground >= level, 1, 0).long()

        if self.accuracy_metric:
            self.accuracy_metric.add(pred, label)

        if self.mioud_metric:
            self.mioud_metric.add(pred, label)

        if self.miouic_metric:
            self.miouic_metric.add(pred, label, image_file)

        if self.miouk_metric:
            self.miouk_metric.add(pred,
                                  label,
                                  panoptic,
                                  image_file,
                                  panoptic_file)

        if self.eced_metric:
            self.eced_metric.add(prob, label)

        if self.ecei_metric:
            self.ecei_metric.add(prob, label)

        if self.sced_metric:
            self.sced_metric.add(prob, label)

        if self.scei_metric:
            self.scei_metric.add(prob, label)


    def value(self):
        results = {}

        if self.accuracy_metric:
            results.update(self.accuracy_metric.value())

        if self.mioud_metric:
            results.update(self.mioud_metric.value())

        if self.miouic_metric:
            results.update(self.miouic_metric.value())

            if self.q:
                results.update(self.miouic_metric.value(self.q))

            if self.q_bar:
                miouicq = {"mIoUIq": 0,
                           "mDiceIq": 0,
                           "mIoUCq": 0,
                           "mDiceCq": 0}

                for q in range(10, 110, 10):
                    miouic_dict = self.miouic_metric.value(q)
                    for key in miouic_dict:
                        key_q = key.replace(f"{q}", "q")
                        miouicq[key_q] += (miouic_dict[key] / 10)

                results.update(miouicq)

        if self.miouk_metric:
            results.update(self.miouk_metric.value())

            if self.q:
                results.update(self.miouk_metric.value(self.q))

            if self.q_bar:
                mioukq = {"mIoUKq": 0,
                          "mDiceKq": 0}

                for q in range(10, 110, 10):
                    miouk_dict = self.miouk_metric.value(q)
                    for key in miouk_dict:
                        key_q = key.replace(f"{q}", "q")
                        mioukq[key_q] += (miouk_dict[key] / 10)

                results.update(mioukq)

        if self.eced_metric:
            results.update(self.eced_metric.value())

        if self.ecei_metric:
            results.update(self.ecei_metric.value())

        if self.sced_metric:
            results.update(self.sced_metric.value())

        if self.scei_metric:
            results.update(self.scei_metric.value())

        return results
