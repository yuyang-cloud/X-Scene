import torch
import torch.distributed as dist

import occgen.utils.constants as C


class Metrics:
    def __init__(self, n_classes, device):
        self.n_classes = n_classes
        self.device = device
        self.tp = None
        self.fp = None
        self.fn = None

        self.otp = None
        self.ofp = None
        self.ofn = None
        self.reset()

    def reset(self):
        self.tp = torch.zeros(self.n_classes, dtype=torch.int64, device=self.device)
        self.fp = torch.zeros(self.n_classes, dtype=torch.int64, device=self.device)
        self.fn = torch.zeros(self.n_classes, dtype=torch.int64, device=self.device)

        self.otp = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.ofp = torch.zeros(1, dtype=torch.int64, device=self.device)
        self.ofn = torch.zeros(1, dtype=torch.int64, device=self.device)

    def get_metrics(self, tp=None, fp=None, fn=None, otp=None, ofp=None, ofn=None, use_mask=False):
        tp = tp if tp is not None else self.tp
        fp = fp if fp is not None else self.fp
        fn = fn if fn is not None else self.fn
        otp = torch.tensor([otp]) if otp is not None else self.otp
        ofp = torch.tensor([ofp]) if ofp is not None else self.ofp
        ofn = torch.tensor([ofn]) if ofn is not None else self.ofn

        union = tp + fp + fn + C.EPSILON
        iou = (tp / union * 100).cpu().numpy()
        if use_mask:
            iou_mask = ((tp + fn) != 0).cpu().numpy()
            miou_all = iou[iou_mask].mean()
            miou_sem = iou[iou_mask][1:].mean()
        else:
            miou_all = iou.mean()
            miou_sem = iou[1:].mean()

        iou = otp / (otp + ofp + ofn + C.EPSILON)
        iou = iou.cpu().numpy()[0] * 100.

        accuracy = tp.sum().item() / (tp.sum() + fp.sum() + C.EPSILON)
        return iou, miou_all, miou_sem, accuracy

    def get_metrics_dist(self):
        tp_all = self.tp.clone()
        fp_all = self.fp.clone()
        fn_all = self.fn.clone()

        dist.all_reduce(tp_all, op=dist.ReduceOp.SUM)
        dist.all_reduce(fp_all, op=dist.ReduceOp.SUM)
        dist.all_reduce(fn_all, op=dist.ReduceOp.SUM)

        return self.get_metrics(tp_all, fp_all, fn_all, use_mask=True)

    def add_batch(self, x, y):
        x = x.reshape(-1)
        y = y.reshape(-1)

        mask = x == y
        tp = torch.bincount(x[mask], minlength=self.n_classes)
        fp = torch.bincount(x, minlength=self.n_classes) - tp
        fn = torch.bincount(y, minlength=self.n_classes) - tp

        self.tp += tp
        self.fp += fp
        self.fn += fn

        x[x != 0] = 1
        y[y != 0] = 1
        omask = x == y
        otp = torch.bincount(x[omask], minlength=2)
        ofp = torch.bincount(x, minlength=2) - otp
        ofn = torch.bincount(y, minlength=2) - otp

        self.otp += otp[1]
        self.ofp += ofp[1]
        self.ofn += ofn[1]

        return self.get_metrics(tp, fp, fn, otp[1], ofp[1], ofn[1])
