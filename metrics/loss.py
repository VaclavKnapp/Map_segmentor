import torch
import torch.nn as nn
from utils.point_utils import extract_foreground_points, iou_loss, chamfer_distance

class ChamferIoULoss(nn.Module):
    """
    Combined loss function using Chamfer distance and IoU.
    
    Args:
        alpha (float): Weight for Chamfer distance component
        beta (float): Weight for IoU component
    """
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred_mask, gt_mask):
        """
        Compute the combined Chamfer-IoU loss.
        
        Args:
            pred_mask (torch.Tensor): Predicted binary mask
            gt_mask (torch.Tensor): Ground truth binary mask
            
        Returns:
            torch.Tensor: Combined loss value
        """
        device = pred_mask.device
        
        # Ensure masks are 2D (squeeze if needed)
        if pred_mask.dim() == 4:
            pred_mask = pred_mask.squeeze(1)  
        if gt_mask.dim() == 4:
            gt_mask = gt_mask.squeeze(1)      
        
        batch_size = pred_mask.shape[0]
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        for b in range(batch_size):
            # Extract points from masks
            pred_points = extract_foreground_points(pred_mask[b])
            gt_points = extract_foreground_points(gt_mask[b])
            
            # Compute Chamfer distance
            cd = chamfer_distance(pred_points, gt_points, device=device)
            
            # Compute IoU loss
            iou = iou_loss(pred_mask[b].bool(), gt_mask[b].bool())
            
            # Combined loss
            total_loss = total_loss + (self.alpha * cd + self.beta * iou)
        
        return total_loss / batch_size
