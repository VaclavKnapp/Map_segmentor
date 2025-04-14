import numpy as np
import torch

class IOUEval:
    """
    Class for evaluating IoU metrics.
    """
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
        self.overall_acc += overall_acc
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

def tversky_index(y_true, y_pred, smooth=1.0, alpha=0.3, beta=0.7):
    """
    Compute Tversky index between two binary masks.
    
    Args:
        y_true (torch.Tensor): Ground truth mask
        y_pred (torch.Tensor): Predicted mask
        smooth (float): Small constant to avoid division by zero
        alpha (float): False negative weight
        beta (float): False positive weight
        
    Returns:
        torch.Tensor: Tversky index
    """
    y_true_flat = y_true.view(-1)
    y_pred_flat = y_pred.view(-1)
    true_pos = torch.sum(y_true_flat * y_pred_flat)
    false_neg = torch.sum(y_true_flat * (1 - y_pred_flat))
    false_pos = torch.sum((1 - y_true_flat) * y_pred_flat)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)

def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    """
    Compute Focal Tversky loss.
    
    Args:
        y_true (torch.Tensor): Ground truth mask
        y_pred (torch.Tensor): Predicted mask
        gamma (float): Focal parameter
        
    Returns:
        torch.Tensor: Focal Tversky loss
    """
    tv = tversky_index(y_true, y_pred)
    return torch.pow((1 - tv), gamma)

def hausdorff_distance(x, y):
    """
    Compute Hausdorff distance between two point sets.
    
    Args:
        x (torch.Tensor): First point set
        y (torch.Tensor): Second point set
        
    Returns:
        torch.Tensor: Hausdorff distance
    """
    pairwise_dist = torch.cdist(x, y, p=2)
    forward_hausdorff = pairwise_dist.min(dim=1)[0].max()
    backward_hausdorff = pairwise_dist.min(dim=0)[0].max()
    hd = torch.max(forward_hausdorff, backward_hausdorff)
    
    return hd
