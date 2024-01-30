from .focal_loss import FocalLoss
from .kl_div_loss import KLDivLoss
from .jdt_loss import JDTLoss


def get_criterion(args):
    if args.label_config.get("HARD", False) or args.label_config.get("KD", False):
        criterion_ce = FocalLoss(ignore_index=args.data_config["num_classes"])
    elif args.label_config.get("LS", False) or args.label_config.get("MR", False):
        criterion_ce = KLDivLoss(T=1)
    else:
        raise NotImplementedError

    if args.label_config.get("HARD", False) or args.label_config.get("LS", False) or args.label_config.get("MR", False):
        criterion_kl = None
    elif args.label_config.get("KD", False):
        criterion_kl = KLDivLoss(T=args.label_config["T"])
    else:
        raise NotImplementedError

    criterion_jdt = JDTLoss(mIoUD=args.loss_config["mIoUD"],
                            mIoUI=args.loss_config["mIoUI"],
                            mIoUC=args.loss_config["mIoUC"],
                            alpha=args.loss_config["alpha"],
                            beta=args.loss_config["beta"],
                            gamma=args.loss_config["gamma"],
                            smooth=args.loss_config["smooth"],
                            threshold=args.loss_config["threshold"],
                            log_loss=args.loss_config["log_loss"],
                            ignore_index=args.data_config["num_classes"],
                            class_weights=args.loss_config["class_weights"],
                            active_classes_mode_hard= \
                                args.loss_config["active_classes_mode_hard"],
                            active_classes_mode_soft= \
                                args.loss_config["active_classes_mode_soft"])

    return criterion_ce, criterion_kl, criterion_jdt
