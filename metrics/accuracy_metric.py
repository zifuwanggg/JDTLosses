import torch
import torch.nn.functional as F

from metrics.metric import Metric


class AccuracyMetric(Metric):
    def __init__(self,
                 accuracyD=True,
                 accuracyI=True,
                 accuracyC=True,
                 q=1,
                 binary=False,
                 num_classes=19,
                 ignore_index=None):
        super().__init__()
        self.accuracyD = accuracyD
        self.accuracyI = accuracyI
        self.accuracyC = accuracyC
        self.q = q
        self.binary = binary
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        self.tp = torch.tensor([])
        self.tn = torch.tensor([])
        self.fp = torch.tensor([])
        self.fn = torch.tensor([])
        self.active_classes = torch.tensor([], dtype=torch.bool)
        self.image_file = []


    def add(self, pred, label, image_file=None):
        batch_size = pred.size(0)

        pred = pred.view(batch_size, -1)
        label = label.view(batch_size, -1)
        keep_mask = (label != self.ignore_index)
        keep_mask = keep_mask.unsqueeze(1).expand(batch_size, self.num_classes, -1)

        pred = F.one_hot(pred, num_classes=self.num_classes).permute(0, 2, 1)
        label = torch.clamp(label, 0, self.num_classes - 1)
        label = F.one_hot(label, num_classes=self.num_classes).permute(0, 2, 1)

        for i in range(batch_size):
            keep_mask_i = keep_mask[i, :, :]
            pred_i = pred[i, :, :][keep_mask_i].reshape(self.num_classes, -1)
            label_i = label[i, :, :][keep_mask_i].reshape(self.num_classes, -1)

            if label_i.size(1) < 1:
                continue

            if image_file == None:
                image_file_i = ""
            else:
                image_file_i = image_file[i]

            tp = torch.logical_and(pred_i == 1, label_i == 1)
            tn = torch.logical_and(pred_i == 0, label_i == 0)
            fp = torch.logical_and(pred_i == 1, label_i == 0)
            fn = torch.logical_and(pred_i == 0, label_i == 1)

            tp = torch.sum(tp, dim=1).unsqueeze(0)
            tn = torch.sum(tn, dim=1).unsqueeze(0)
            fp = torch.sum(fp, dim=1).unsqueeze(0)
            fn = torch.sum(fn, dim=1).unsqueeze(0)

            if self.binary:
                mask = torch.amax(pred_i + label_i, dim=1) > 0.5
            else:
                mask = torch.amax(label_i, dim=1) > 0.5
            mask = mask.unsqueeze(0)
            active_classes = torch.zeros(self.num_classes, dtype=torch.bool).unsqueeze(0)
            active_classes[mask] = 1

            self.tp = torch.cat((self.tp, tp), dim=0)
            self.tn = torch.cat((self.tn, tn), dim=0)
            self.fp = torch.cat((self.fp, fp), dim=0)
            self.fn = torch.cat((self.fn, fn), dim=0)
            self.active_classes = torch.cat((self.active_classes, active_classes), dim=0)
            self.image_file.append(image_file_i)

    def value(self):
        results = {}

        if self.accuracyD:
            results.update(self.valueD())

        if self.accuracyI:
            results.update(self.valueI())

        if self.accuracyC:
            results.update(self.valueC())

        return results


    def valueD(self):
        tp = torch.sum(self.tp, dim=0)
        fp = torch.sum(self.fp, dim=0)
        fn = torch.sum(self.fn, dim=0)

        if self.binary:
            tp = tp[1]
            fp = fp[1]
            fn = fn[1]

        Acc = 100 * torch.sum(tp) / torch.sum(tp + fn)
        mAccD = 100 * torch.mean(tp / (tp + fn))
        mIoUD = 100 * torch.mean(tp / (tp + fp + fn))
        mDiceD = 100 * torch.mean(2 * tp / (2 * tp + fp + fn))

        return {"Acc": Acc,
                "mAccD": mAccD,
                "mIoUD": mIoUD,
                "mDiceD": mDiceD}


    def valueI(self):
        AccIC = self.tp / (self.tp + self.fn)
        IoUIC = self.tp / (self.tp + self.fp + self.fn)
        DiceIC = 2 * self.tp / (2 * self.tp + self.fp + self.fn)

        AccIC[~self.active_classes] = 0
        IoUIC[~self.active_classes] = 0
        DiceIC[~self.active_classes] = 0

        mAccI = self.reduceI(AccIC)
        mIoUI = self.reduceI(IoUIC)
        mDiceI = self.reduceI(DiceIC)

        mAccIQ = mIoUIQ = mDiceIQ = 0
        for q in range(10, 110, 10):
            mAccIQ += self.reduceI(AccIC, q)
            mIoUIQ += self.reduceI(IoUIC, q)
            mDiceIQ += self.reduceI(DiceIC, q)
        mAccIQ /= 10
        mIoUIQ /= 10
        mDiceIQ /= 10

        mAccIq = self.reduceI(AccIC, self.q)
        mIoUIq = self.reduceI(IoUIC, self.q)
        mDiceIq = self.reduceI(DiceIC, self.q)

        return {"mAccI": mAccI,
                "mIoUI": mIoUI,
                "mDiceI": mDiceI,
                "mAccIQ": mAccIQ,
                "mIoUIQ": mIoUIQ,
                "mDiceIQ": mDiceIQ,
                f"mAccI{self.q}": mAccIq,
                f"mIoUI{self.q}": mIoUIq,
                f"mDiceI{self.q}": mDiceIq}


    def valueC(self):
        AccIC = self.tp / (self.tp + self.fn)
        IoUIC = self.tp / (self.tp + self.fp + self.fn)
        DiceIC = 2 * self.tp / (2 * self.tp + self.fp + self.fn)

        AccIC[~self.active_classes] = 1e6
        IoUIC[~self.active_classes] = 1e6
        DiceIC[~self.active_classes] = 1e6

        mAccC = self.reduceC(AccIC)
        mIoUC = self.reduceC(IoUIC)
        mDiceC = self.reduceC(DiceIC)

        mAccCQ = mIoUCQ = mDiceCQ = 0
        for q in range(10, 110, 10):
            mAccCQ += self.reduceC(AccIC, q)
            mIoUCQ += self.reduceC(IoUIC, q)
            mDiceCQ += self.reduceC(DiceIC, q)
        mAccCQ /= 10
        mIoUCQ /= 10
        mDiceCQ /= 10

        mAccCq = self.reduceC(AccIC, self.q)
        mIoUCq = self.reduceC(IoUIC, self.q)
        mDiceCq = self.reduceC(DiceIC, self.q)

        return {"mAccC": mAccC,
                "mIoUC": mIoUC,
                "mDiceC": mDiceC,
                "mAccCQ": mAccCQ,
                "mIoUCQ": mIoUCQ,
                "mDiceCQ": mDiceCQ,
                f"mAccC{self.q}": mAccCq,
                f"mIoUC{self.q}": mIoUCq,
                f"mDiceC{self.q}": mDiceCq}


    def reduceI(self, value_matrix, q=None):
        active_sum = torch.sum(self.active_classes, dim=1)
        if self.binary:
            value = value_matrix[:, 1]
            value[active_sum < 2] = 1
        else:
            value = torch.sum(value_matrix, dim=1)
            value /= active_sum

        if q == None:
            n = value.size(0)
        else:
            n = max(1, int(q / 100 * value.size(0)))

        value = torch.sort(value)[0][:n]
        value = 100 * torch.mean(value)

        return value


    def reduceC(self, value_matrix, q=None):
        num_images, num_classes = value_matrix.shape
        active_sum = torch.sum(self.active_classes, dim=0)
        if q != None:
            active_sum = torch.max(torch.ones(num_classes), (q / 100 * active_sum).to(torch.long))

        indices = torch.arange(num_images).view(-1, 1).expand_as(value_matrix)
        mask = indices < active_sum

        value_matrix = mask * torch.sort(value_matrix, dim=0)[0]
        value = torch.sum(value_matrix, dim=0) / active_sum
        value = 100 * torch.mean(value)

        return value
