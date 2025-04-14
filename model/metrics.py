import numpy as np
import torch
import torch.nn.functional as F

class IOUEval:
    def __init__(self, nClasses):
        self.nClasses = nClasses
        self.reset()
    def reset(self):
        self.overall_acc = 0
        self.per_class_acc = np.zeros(self.nClasses, dtype=np.float32)
        self.per_class_iu = np.zeros(self.nClasses, dtype=np.float32)
        self.mIOU = 0
        self.batchCount = 1
    def fast_hist(self, a, b):
        k = (a >= 0) & (a < self.nClasses)
        return np.bincount(self.nClasses * a[k].astype(int) + b[k], minlength=self.nClasses ** 2).reshape(self.nClasses, self.nClasses)
    def compute_hist(self, predict, gth):
        hist = self.fast_hist(gth, predict)
        return hist
    def addBatch(self, predict, gth):
        predict = predict.cpu().numpy().flatten()
        gth = gth.cpu().numpy().flatten()
        epsilon = 0.00000001
        hist = self.compute_hist(predict, gth)
        overall_acc = np.diag(hist).sum() / (hist.sum() + epsilon)
        per_class_acc = np.diag(hist) / (hist.sum(1) + epsilon)
        per_class_iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)
        mIou = np.nanmean(per_class_iu)
        self.overall_acc +=overall_acc
        self.per_class_acc += per_class_acc
        self.per_class_iu += per_class_iu
        self.mIOU += mIou
        self.batchCount += 1
    def getMetric(self):
        overall_acc = self.overall_acc/self.batchCount
        per_class_acc = self.per_class_acc / self.batchCount
        per_class_iu = self.per_class_iu / self.batchCount
        mIOU = self.mIOU / self.batchCount
        return overall_acc, per_class_acc, per_class_iu, mIOU

def chamfer_loss(pred_points, target_points):

    dist_matrix = torch.cdist(pred_points.float(), target_points.float(), p=2)  # Shape: [N_p, N_t]


    min_dist_p2t, _ = torch.min(dist_matrix, dim=1)  # Shape: [N_p]

    min_dist_t2p, _ = torch.min(dist_matrix, dim=0)  # Shape: [N_t]


    loss = (min_dist_p2t ** 2).mean() + (min_dist_t2p ** 2).mean()
    return loss

def tversky_index(y_true, y_pred, smooth=1.0, alpha=0.3, beta=0.7):
    y_true_flat = y_true.view(-1)
    y_pred_flat = y_pred.view(-1)
    true_pos = torch.sum(y_true_flat * y_pred_flat)
    false_neg = torch.sum(y_true_flat * (1 - y_pred_flat))
    false_pos = torch.sum((1 - y_true_flat) * y_pred_flat)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)

def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky_index(y_true, y_pred)
    return torch.pow((1 - tv), gamma)

def hausdorff_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    pairwise_dist = torch.cdist(x, y, p=2)
    forward_hausdorff = pairwise_dist.min(dim=1)[0].max()
    backward_hausdorff = pairwise_dist.min(dim=0)[0].max()
    hd = torch.max(forward_hausdorff, backward_hausdorff)
    
    return hd


