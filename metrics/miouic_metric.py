import torch
import torch.nn.functional as F

from .metric import Metric


class mIoUICMetric(Metric):
    def __init__(self, num_classes, ignore_index=None):
        super().__init__()
        self.IoUI = []
        self.DiceI = []
        self.IoUC = [[] for _ in range(num_classes)]
        self.DiceC = [[] for _ in range(num_classes)]

        self.num_classes = num_classes
        self.ignore_index = ignore_index


    def add(self, pred, label, image_file=None):
        batch_size = pred.size(0)

        pred = pred.view(batch_size, -1)
        label = label.view(batch_size, -1)
        keep_mask = (label != self.ignore_index). \
            unsqueeze(0).expand(self.num_classes, batch_size, -1)

        pred = F.one_hot(
            pred, num_classes=self.num_classes). \
                permute(2, 0, 1)
        label = F.one_hot(
            torch.clamp(label, 0, self.num_classes - 1), num_classes=self.num_classes). \
                permute(2, 0, 1)

        for i in range(batch_size):
            keep_mask_i = keep_mask[:, i, :]
            pred_i = pred[:, i, :][keep_mask_i].reshape(self.num_classes, -1)
            label_i = label[:, i, :][keep_mask_i].reshape(self.num_classes, -1)

            if image_file is None:
                image_file_i = ""
            else:
                image_file_i = image_file[i]

            if label_i.size(1) < 1:
                continue

            all_classes = torch.arange(self.num_classes)
            if self.num_classes == 2:
                classes = all_classes[torch.amax(pred_i + label_i, dim=1) > 0.5]
                if classes.size(0) == 2:
                    classes = classes[1:]
                else:
                    self.IoUI.append((1, image_file_i))
                    self.DiceI.append((1, image_file_i))
                    self.IoUC[1].append((1, image_file_i))
                    self.DiceC[1].append((1, image_file_i))
                    continue
            else:
                classes = all_classes[torch.amax(label_i, dim=1) > 0.5]

            cardinality = torch.sum(pred_i + label_i, dim=1)
            difference = torch.sum(torch.abs(pred_i - label_i), dim=1)
            intersection = (cardinality - difference) / 2
            union = (cardinality + difference) / 2

            IoU = intersection / union
            Dice = 2 * intersection / cardinality

            self.IoUI.append((IoU[classes].mean().item(), image_file_i))
            self.DiceI.append((Dice[classes].mean().item(), image_file_i))

            for c in classes:
                self.IoUC[c].append((IoU[c].item(), image_file_i))
                self.DiceC[c].append((Dice[c].item(), image_file_i))


    def value(self, q=""):
        if q == "":
            n = len(self.IoUI)
        else:
            n = max(1, int(q / 100 * len(self.IoUI)))

        self.IoUI.sort()
        self.DiceI.sort()

        IoUI = [self.IoUI[i][0] for i in range(n)]
        DiceI = [self.DiceI[i][0] for i in range(n)]

        mIoUI = 100 * sum(IoUI) / len(IoUI)
        mDiceI = 100 * sum(DiceI) / len(DiceI)

        if self.num_classes == 2:
            return mIoUI, mDiceI, mIoUI, mDiceI

        IoUC = []
        DiceC = []

        for c in range(self.num_classes):
            if len(self.IoUC[c]) < 1:
                continue

            if q == "":
                n = len(self.IoUC[c])
            else:
                n = max(1, int(q / 100 * len(self.IoUC[c])))

            self.IoUC[c].sort()
            self.DiceC[c].sort()

            IoUc = [self.IoUC[c][i][0] for i in range(n)]
            Dicec = [self.DiceC[c][i][0] for i in range(n)]

            IoUC.append(sum(IoUc) / len(IoUc))
            DiceC.append(sum(Dicec) / len(Dicec))

        mIoUC = 100 * sum(IoUC) / len(IoUC)
        mDiceC = 100 * sum(DiceC) / len(DiceC)

        return {f"mIoUI{q}": mIoUI,
                f"mDiceI{q}": mDiceI,
                f"mIoUC{q}": mIoUC,
                f"mDiceC{q}": mDiceC}
