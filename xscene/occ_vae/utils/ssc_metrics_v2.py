import torch

class SSCMetrics_v2:
    def __init__(self, n_classes, ignore=None):
        # classes
        self.n_classes = n_classes

        # What to include and ignore from the means
        self.ignore = torch.tensor(ignore, dtype=torch.int64).cuda()
        self.include = torch.tensor([n for n in range(self.n_classes) if n not in self.ignore], dtype=torch.int64).cuda()

        # reset the class counters
        self.reset()

    def num_classes(self):
        return self.n_classes

    def get_eval_mask(self, labels, invalid_voxels):
        """
        Ignore labels set to 255 and invalid voxels (the ones never hit by a laser ray, probed using ray tracing)
        :param labels: input ground truth voxels
        :param invalid_voxels: voxels ignored during evaluation since the lie beyond the scene that was captured by the laser
        :return: boolean mask to subsample the voxels to evaluate
        """
        masks = torch.ones_like(labels, dtype=torch.bool)
        masks[labels == 255] = False
        masks[invalid_voxels == 1] = False
        return masks

    def reset(self):
        self.conf_matrix = torch.zeros((self.n_classes, self.n_classes), dtype=torch.int64).cuda()
        
    def one_stats(self, x, y):
        x_row = x.view(-1)  # de-batchify
        y_row = y.view(-1)  # de-batchify

        idxs = torch.stack((x_row, y_row), dim=0)
        conf_matrix = torch.zeros((self.n_classes, self.n_classes), dtype=torch.int64).cuda()
        conf_matrix.index_put_(tuple(idxs), torch.tensor(1), accumulate=True)

        conf_matrix[:, self.ignore] = 0

        tp = torch.diag(conf_matrix)
        fp = conf_matrix.sum(dim=1) - tp
        fn = conf_matrix.sum(dim=0) - tp

        intersection = tp
        union = tp + fp + fn + 1e-15

        n = len(torch.unique(y)) - 1

        miou = (intersection[1:] / union[1:]).sum() / n * 100
        all_miou = (intersection / union).sum() / (n + 1) * 100

        iou = (torch.sum(conf_matrix[1:, 1:])) / (torch.sum(conf_matrix) - conf_matrix[0, 0] + 1e-8) * 100

        return iou, miou, all_miou
    
    def addBatch(self, x, y):  # x=preds, y=targets
        # sizes should be matching
        x_row = x.view(-1)  # de-batchify
        y_row = y.view(-1)  # de-batchify

        # check
        assert(x_row.shape == y_row.shape)

        # create indexes
        idxs = torch.stack((x_row, y_row), dim=0)

        # make confusion matrix (cols = gt, rows = pred)
        self.conf_matrix.index_put_(tuple(idxs), torch.tensor(1), accumulate=True)
        iou, miou, all_miou = self.one_stats(x, y)
        return iou, miou
        

    def getStats(self):
        # remove fp from confusion on the ignore classes cols
        conf = self.conf_matrix.clone()
        conf[:, self.ignore] = 0

        # get the clean stats
        tp = torch.diag(conf)
        fp = conf.sum(dim=1) - tp
        fn = conf.sum(dim=0) - tp
        return tp, fp, fn

    def getIoU(self):
        tp, fp, fn = self.getStats()
        intersection = tp
        union = tp + fp + fn + 1e-15
        iou = intersection / union
        iou_mean = (intersection[self.include] / union[self.include]).mean().item()
        return iou_mean, iou  # returns "iou mean", "iou per class" ALL CLASSES

    def getacc(self):
        tp, fp, fn = self.getStats()
        total_tp = tp.sum().item()
        total = tp[self.include].sum().item() + fp[self.include].sum().item() + 1e-15
        acc_mean = total_tp / total
        return acc_mean  # returns "acc mean"
        
    def get_confusion(self):
        return self.conf_matrix.clone()