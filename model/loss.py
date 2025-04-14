import torch
import torch.nn as nn

def extract_foreground_points(mask):

    coords = mask.nonzero(as_tuple=False)  # (N, 2) => [row_index, col_index]
    return coords.float()

def chamfer_distance_mask(pred_mask, gt_mask, device='cpu'):

    P = extract_foreground_points(pred_mask)
    G = extract_foreground_points(gt_mask)


    if P.size(0) == 0 and G.size(0) == 0:

        return torch.tensor(0.0, device=device, requires_grad=True)
    if P.size(0) == 0 or G.size(0) == 0:

        return torch.tensor(float(P.size(0) + G.size(0)), device=device, requires_grad=True)

    P_expanded = P.unsqueeze(1)  
    G_expanded = G.unsqueeze(0)  
    diffs = P_expanded - G_expanded  
    dist_sq = diffs.pow(2).sum(-1)   


    min_p_to_g, _ = torch.min(dist_sq, dim=1)

    min_g_to_p, _ = torch.min(dist_sq, dim=0)

    cd = torch.sum(min_p_to_g) + torch.sum(min_g_to_p)
    return cd

def iou_loss(pred_mask, gt_mask, eps=1e-6):

    intersection = (pred_mask & gt_mask).float().sum()
    union        = (pred_mask | gt_mask).float().sum() + eps
    iou = intersection / union
    return 1.0 - iou

class ChamferIoULoss(nn.Module):

    def __init__(self, alpha=1.0, beta=1.0):

        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred_mask, gt_mask):

        device = pred_mask.device
        

        if pred_mask.dim() == 4:
            pred_mask = pred_mask.squeeze(1)  
        if gt_mask.dim() == 4:
            gt_mask = gt_mask.squeeze(1)      
        
        batch_size = pred_mask.shape[0]
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        for b in range(batch_size):
            cd = chamfer_distance_mask(pred_mask[b], gt_mask[b], device=device)
            iou = iou_loss(pred_mask[b], gt_mask[b])
            total_loss = total_loss + (self.alpha * cd + self.beta * iou)
        
        return total_loss / batch_size
