import sys
import os
import torch
import torch.nn as nn
from tqdm import tqdm

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics.metrics import IOUEval
from utils.visualization import draw_translucent_seg_maps
from metrics.loss import ChamferIoULoss

def train(
    model,
    train_dataloader,
    device,
    optimizer,
    classes_to_train,
    alpha=1.0,
    beta=1.0
):
    """
    Training function.
    
    Args:
        model: The model to train
        train_dataloader: DataLoader for training data
        device: Device to use for training
        optimizer: Optimizer for training
        classes_to_train: List of classes to train
        alpha: Weight for Chamfer distance in loss
        beta: Weight for IoU in loss
        
    Returns:
        tuple: Training loss, accuracy, and mean IoU
    """
    print('Training')
    model.train()
    train_running_loss = 0.0
    prog_bar = tqdm(
        train_dataloader, 
        total=len(train_dataloader), 
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
    )
    counter = 0  # To keep track of batch counter
    num_classes = len(classes_to_train)
    iou_eval = IOUEval(num_classes)
    
    # Initialize the loss function
    chamfer_iou_loss = ChamferIoULoss(alpha=alpha, beta=beta).to(device)

    for i, data in enumerate(prog_bar):
        counter += 1
        pixel_values, target = data['pixel_values'].to(device), data['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values)

        logits = outputs.logits  # Shape: [batch_size, num_classes, H, W]
        upsampled_logits = nn.functional.interpolate(
            logits, size=target.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        predicted = upsampled_logits.argmax(dim=1)  # Shape: [batch_size, H, W]
        
        # Convert logits to binary masks for loss calculation
        predicted_masks = (predicted == 1).float()  # River class has index 1
        target_masks = (target == 1).float()
        
        # Calculate loss using ChamferIoULoss
        loss = chamfer_iou_loss(predicted_masks, target_masks)
        train_running_loss += loss.item()

        # Backpropagation and parameter update
        loss.backward()
        optimizer.step()

        # Update IOU metrics
        iou_eval.addBatch(predicted.data, target.data)

    # Compute per-epoch loss and metrics
    train_loss = train_running_loss / counter
    overall_acc, per_class_acc, per_class_iou, mIOU = iou_eval.getMetric()
    return train_loss, overall_acc, mIOU


def validate(
    model,
    valid_dataloader,
    device,
    classes_to_train,
    label_colors_list,
    epoch,
    save_dir,
    alpha=1.0,
    beta=1.0,
    viz_map=None
):
    """
    Validation function.
    
    Args:
        model: The model to validate
        valid_dataloader: DataLoader for validation data
        device: Device to use for validation
        classes_to_train: List of classes to train
        label_colors_list: List of RGB colors for each class
        epoch: Current epoch number
        save_dir: Directory to save visualization
        alpha: Weight for Chamfer distance in loss
        beta: Weight for IoU in loss
        viz_map: Visualization color map
        
    Returns:
        tuple: Validation loss, accuracy, and mean IoU
    """
    print('Validating')
    model.eval()
    valid_running_loss = 0.0
    num_classes = len(classes_to_train)
    iou_eval = IOUEval(num_classes)
    
    # Initialize the loss function
    chamfer_iou_loss = ChamferIoULoss(alpha=alpha, beta=beta).to(device)
    
    # Use default viz_map if None is provided
    if viz_map is None:
        viz_map = label_colors_list

    with torch.no_grad():
        prog_bar = tqdm(
            valid_dataloader, 
            total=(len(valid_dataloader)), 
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
        )
        counter = 0 # To keep track of batch counter.
        for i, data in enumerate(prog_bar):
            counter += 1
            pixel_values, target = data['pixel_values'].to(device), data['labels'].to(device)
            outputs = model(pixel_values=pixel_values)

            logits = outputs.logits
            upsampled_logits = nn.functional.interpolate(
                logits, size=target.shape[-2:], 
                mode="bilinear", 
                align_corners=False
            )
            
            predicted = upsampled_logits.argmax(dim=1)
            
            # Save the validation segmentation maps
            if i == 1:
                draw_translucent_seg_maps(
                    pixel_values, 
                    upsampled_logits, 
                    epoch, 
                    i, 
                    save_dir, 
                    label_colors_list,
                    viz_map
                )

            # Convert predictions to binary masks for loss calculation
            predicted_masks = (predicted == 1).float()  # River class has index 1
            target_masks = (target == 1).float()
            
            # Calculate loss using ChamferIoULoss
            loss = chamfer_iou_loss(predicted_masks, target_masks)
            valid_running_loss += loss.item()

            iou_eval.addBatch(predicted.data, target.data)
        
    # Compute per-epoch loss and metrics
    valid_loss = valid_running_loss / counter
    overall_acc, per_class_acc, per_class_iou, mIOU = iou_eval.getMetric()
    return valid_loss, overall_acc, mIOU
