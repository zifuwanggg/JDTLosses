import torch
import torch.nn as nn


def get_idx_idy(label, num_keys, biased, ignore_index):
    classes, counts = label.unique(return_counts=True)

    if ignore_index in classes:
        classes = classes[:-1]
        counts = counts[:-1]

    assert ignore_index not in classes

    if biased:
        num_samples = torch.ceil(num_keys * counts / counts.sum())
    else:
        num_samples = torch.ceil(num_keys * torch.ones_like(classes) / len(classes))

    for j, (n, c, ctn) in enumerate(zip(num_samples, classes, counts)):
        weights = ((label == c).to(torch.float32)).reshape(-1)
        index = torch.multinomial(
            input=weights, num_samples=int(min(n, ctn)), replacement=False)
        idx_label, idy_label = torch.div(index, label.shape[0], rounding_mode='floor'), torch.remainder(index, label.shape[0])

        if j == 0:
            idx = idx_label
            idy = idy_label
        else:
            idx = torch.cat((idx, idx_label))
            idy = torch.cat((idy, idy_label))

    return idx, idy


def get_kde(prob_flatten,
            prob_key,
            label_key_one_hot,
            bandwidth,
            crop_size):
    """
    https://github.com/tpopordanoska/ece-kde/blob/main/ece_kde.py
    This function was only tested on LiTS/KiTS
    It could be numerically unstable on datasets with a large number of classes
    """
    alphas = (prob_key / bandwidth + 1).T
    log_beta = (torch.sum((torch.lgamma(alphas)), dim=1) - torch.lgamma(torch.sum(alphas, dim=1)))
    log_num = torch.matmul(torch.log(prob_flatten), (alphas-1).T)
    log_dir_pdf = log_num - log_beta

    kernel = torch.exp(log_dir_pdf)
    kernel_y = torch.matmul(kernel, label_key_one_hot)

    den = torch.sum(kernel, dim=1)
    den = torch.clamp(den, min=1e-10)
    ratio = kernel_y / den.unsqueeze(-1)
    prob = torch.permute(ratio.reshape(crop_size[0], crop_size[1], -1), (2, 0, 1))

    return prob


def kde(prob, aux_prob, label, label_boundary, args):
    for i in range(label.shape[0]):
        prob_i = prob[i, ]
        aux_prob_i = aux_prob[i, ]
        label_i = label[i, ]
        label_boundary_i = label_boundary[i,]

        pred_i = prob_i.argmax(0)
        misprediction_i = pred_i != label_i
        mask_i = torch.logical_or(misprediction_i, label_boundary_i)

        idx, idy = get_idx_idy(label_i,
                               args.train_config["KD"]["kde"]["num_keys"],
                               args.train_config["KD"]["kde"]["biased"],
                               args.data_config["num_classes"])

        prob_flatten_i = prob_i.reshape(prob_i.shape[0], -1).T
        aux_prob_flatten_i = aux_prob_i.reshape(aux_prob_i.shape[0], -1).T
        prob_key_i = prob_i[:, idx, idy]
        aux_prob_key_i = aux_prob_i[:, idx, idy]
        label_key_i = label_i[idx, idy]
        label_key_one_hot_i = nn.functional.one_hot(
            label_key_i, num_classes=args.data_config["num_classes"]).to(torch.float32)

        prob[i, ][mask_i] = get_kde(prob_flatten_i,
                                    prob_key_i,
                                    label_key_one_hot_i,
                                    args.train_config["KD"]["kde"]["bandwidth"],
                                    args.data_config["crop_size"])[mask_i]
        aux_prob[i, ][mask_i] = get_kde(aux_prob_flatten_i,
                                        aux_prob_key_i,
                                        label_key_one_hot_i,
                                        args.train_config["KD"]["kde"]["bandwidth"],
                                        args.data_config["crop_size"])[mask_i]

    return prob, aux_prob
