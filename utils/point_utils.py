import torch
import numpy as np

def extract_foreground_points(mask):
    """
    Extract coordinates of foreground pixels from a binary mask.
    
    Args:
        mask (torch.Tensor): Binary mask tensor
        
    Returns:
        torch.Tensor: Coordinates of foreground pixels as float tensor of shape [N, 2]
    """
    coords = mask.nonzero(as_tuple=False)  # (N, 2) => [row_index, col_index]
    return coords.float()

def chamfer_distance(pred_points, target_points, device='cpu'):
    """
    Compute the Chamfer distance between two point sets.
    
    Args:
        pred_points (torch.Tensor): First point set of shape [N_p, 2]
        target_points (torch.Tensor): Second point set of shape [N_t, 2]
        device (str): Device to use for computation
        
    Returns:
        torch.Tensor: Chamfer distance
    """
    # Handle empty point sets
    if pred_points.size(0) == 0 and target_points.size(0) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    if pred_points.size(0) == 0 or target_points.size(0) == 0:
        return torch.tensor(float(pred_points.size(0) + target_points.size(0)), 
                            device=device, requires_grad=True)

    # Compute pairwise distance between predicted and target points
    dist_matrix = torch.cdist(pred_points.float(), target_points.float(), p=2)  # Shape: [N_p, N_t]

    # For each point in pred_points, find the min distance to target_points
    min_dist_p2t, _ = torch.min(dist_matrix, dim=1)  # Shape: [N_p]
    # For each point in target_points, find the min distance to pred_points
    min_dist_t2p, _ = torch.min(dist_matrix, dim=0)  # Shape: [N_t]

    # Sum the distances
    loss = (min_dist_p2t ** 2).mean() + (min_dist_t2p ** 2).mean()
    return loss

def iou_loss(pred_mask, gt_mask, eps=1e-6):
    """
    Compute the IoU loss between two binary masks.
    
    Args:
        pred_mask (torch.Tensor): Predicted binary mask
        gt_mask (torch.Tensor): Ground truth binary mask
        eps (float): Small constant to avoid division by zero
        
    Returns:
        torch.Tensor: 1 - IoU (loss value)
    """
    intersection = (pred_mask & gt_mask).float().sum()
    union = (pred_mask | gt_mask).float().sum() + eps
    iou = intersection / union
    return 1.0 - iou

def set_class_values(all_classes, classes_to_train):
    """
    Assign specific class labels to each of the classes.
    For example, `background=0`, `river=1`, and so on.
    
    Args:
        all_classes (list): List containing all class names
        classes_to_train (list): List containing class names to train
        
    Returns:
        list: List of class indices for classes to train
    """
    class_values = [all_classes.index(cls.lower()) for cls in classes_to_train]
    return class_values

def get_label_mask(mask, class_values, label_colors_list):
    """
    Encode pixels belonging to the same class in the image into the same label.
    
    Args:
        mask (numpy.ndarray): Segmentation mask of shape [H, W, 3]
        class_values (list): List of class indices
        label_colors_list (list): List of RGB colors for each class
        
    Returns:
        numpy.ndarray: Label mask of shape [H, W]
    """
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for value in class_values:
        for ii, label in enumerate(label_colors_list):
            if value == label_colors_list.index(label):
                label = np.array(label)
                label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = value
    label_mask = label_mask.astype(int)
    return label_mask
