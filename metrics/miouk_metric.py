import torch
import torch.nn.functional as F

from .metric import Metric


class mIoUKMetric(Metric):
    def __init__(self,
                 annos,
                 anno_map,
                 num_classes,
                 ignore_index=None,
                 reduce_panoptic_zero_label=False,
                 threshold=0):
        super().__init__()

        self.annos = annos
        self.anno_map = anno_map
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reduce_panoptic_zero_label = reduce_panoptic_zero_label
        self.threshold = threshold

        self.IoUK = [[] for _ in range(num_classes)]
        self.DiceK = [[] for _ in range(num_classes)]


    def add(self,
            pred,
            label,
            panoptic,
            image_file,
            panoptic_file):
        batch_size = pred.size(0)

        pred = pred.view(batch_size, -1)
        label = label.view(batch_size, -1)
        panoptic = panoptic.view(batch_size, -1)
        keep_mask = (label != self.ignore_index). \
            unsqueeze(0).expand(self.num_classes, batch_size, -1)

        pred = F.one_hot(pred, num_classes=self.num_classes).permute(2, 0, 1)
        label = F.one_hot(
            torch.clamp(label, 0, self.num_classes - 1), num_classes=self.num_classes). \
                permute(2, 0, 1)

        for i in range(batch_size):
            keep_mask_i = keep_mask[:, i, :]
            pred_i = pred[:, i, :][keep_mask_i].view(self.num_classes, -1)
            label_i = label[:, i, :][keep_mask_i].view(self.num_classes, -1)
            panoptic_i = panoptic[i, :][keep_mask_i[0, :]]

            if image_file is None:
                image_file_i = ""
            else:
                image_file_i = image_file[i]

            panoptic_file_i = panoptic_file[i].split("/")[-1]

            if pred_i.size(1) < 1:
                continue

            index = self.anno_map[panoptic_file_i]
            anno = self.annos[index]
            segments = anno["segments_info"]

            assert anno["file_name"] == panoptic_file_i

            cardinality = torch.sum(pred_i + label_i, dim=1)
            difference = torch.sum(torch.abs(pred_i - label_i), dim=1)
            intersection = 1 / 2 * (cardinality - difference)

            all_area = label_i.sum(dim=1)
            all_fp = torch.sum(pred_i, dim=1) - intersection

            for seg in segments:
                category_id = seg["category_id"] - self.reduce_panoptic_zero_label

                if "ADE_val_00000899" in image_file_i and category_id == 14:
                    continue

                pred_c = pred_i[category_id]
                label_k = (panoptic_i == seg["id"]).float()

                area_c = all_area[category_id]
                area_k = torch.sum(label_k)

                if area_k < self.threshold:
                    continue

                cardinality_k = torch.sum(pred_c + label_k)
                difference_k = torch.sum(torch.abs(pred_c - label_k))
                intersection_k = 1 / 2 * (cardinality_k - difference_k)

                fn_k = area_k - intersection_k
                fp_k = area_k / area_c * all_fp[category_id]

                IoU = intersection_k / (intersection_k + fn_k + fp_k)
                Dice = 2 * intersection_k / (2 * intersection_k + fn_k + fp_k)

                self.IoUK[category_id].append((IoU.item(), image_file_i, seg["id"]))
                self.DiceK[category_id].append((Dice.item(), image_file_i, seg["id"]))


    def value(self, q=""):
        IoUK = []
        DiceK = []

        for c in range(self.num_classes):
            if len(self.IoUK[c]) < 1:
                continue

            if q == "":
                n = len(self.IoUK[c])
            else:
                n = max(1, int(q / 100 * len(self.IoUK[c])))

            self.IoUK[c].sort()
            self.DiceK[c].sort()

            IoUc = [self.IoUK[c][i][0] for i in range(n)]
            Dicec = [self.DiceK[c][i][0] for i in range(n)]

            IoUK.append(sum(IoUc) / len(IoUc))
            DiceK.append(sum(Dicec) / len(Dicec))

        mIoUK = 100 * sum(IoUK) / len(IoUK)
        mDiceK = 100 * sum(DiceK) / len(DiceK)

        return {f"mIoUK{q}": mIoUK,
                f"mDiceK{q}": mDiceK}
