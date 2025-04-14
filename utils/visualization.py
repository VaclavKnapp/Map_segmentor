import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from PIL import Image

def denormalize(x, mean=None, std=None):
    """
    Denormalize an image for visualization.
    
    Args:
        x (numpy.ndarray): Normalized image of shape [H, W, C]
        mean (list): Mean values for each channel
        std (list): Standard deviation values for each channel
        
    Returns:
        numpy.ndarray: Denormalized image of shape [H, W, C]
    """
    # x should be a Numpy array of shape [H, W, C] 
    x = torch.tensor(x).permute(2, 0, 1).unsqueeze(0)
    for t, m, s in zip(x, mean, std):
        t.mul_(s).add_(m)
    res = torch.clamp(t, 0, 1)
    res = res.squeeze(0).permute(1, 2, 0).numpy()
    return res

def draw_translucent_seg_maps(
    data, 
    output, 
    epoch, 
    i, 
    val_seg_dir, 
    label_colors_list,
    viz_map
):
    """
    Draw translucent segmentation maps over original images.
    
    Args:
        data (torch.Tensor): Batch of images
        output (torch.Tensor): Model output logits
        epoch (int): Current epoch
        i (int): Batch index
        val_seg_dir (str): Directory to save visualizations
        label_colors_list (list): List of RGB colors for each class
        viz_map (list): List of RGB colors for visualization
    """
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]
    alpha = 1 # how much transparency
    beta = 0.8 # alpha + beta should be 1
    gamma = 0 # contrast
    seg_map = output[0] # use only one output from the batch
    seg_map = torch.argmax(seg_map.squeeze(), dim=0).detach().cpu().numpy()
    image = denormalize(data[0].permute(1, 2, 0).cpu().numpy(), IMG_MEAN, IMG_STD)
    red_map = np.zeros_like(seg_map).astype(np.uint8)
    green_map = np.zeros_like(seg_map).astype(np.uint8)
    blue_map = np.zeros_like(seg_map).astype(np.uint8)
    for label_num in range(0, len(label_colors_list)):
        index = seg_map == label_num
        red_map[index] = np.array(viz_map)[label_num, 0]
        green_map[index] = np.array(viz_map)[label_num, 1]
        blue_map[index] = np.array(viz_map)[label_num, 2]
        
    rgb = np.stack([red_map, green_map, blue_map], axis=2)
    rgb = np.array(rgb, dtype=np.float32)
    # convert color to BGR format for OpenCV
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) * 255.
    cv2.addWeighted(image, alpha, rgb, beta, gamma, image)
    cv2.imwrite(f"{val_seg_dir}/e{epoch}_b{i}.jpg", image)

def draw_segmentation_map(labels, palette):
    """
    Create a colored segmentation map from label indices.
    
    Args:
        labels (numpy.ndarray): Label indices of shape [H, W]
        palette (list): List of RGB colors for each class
        
    Returns:
        numpy.ndarray: Colored segmentation map of shape [H, W, 3]
    """
    # create Numpy arrays containing zeros
    # later to be used to fill them with respective red, green, and blue pixels
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)
    for label_num in range(0, len(palette)):
        index = labels == label_num
        red_map[index] = np.array(palette)[label_num, 0]
        green_map[index] = np.array(palette)[label_num, 1]
        blue_map[index] = np.array(palette)[label_num, 2]
        
    segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
    return segmentation_map

def image_overlay(image, segmented_image):
    """
    Overlay segmentation map on original image.
    
    Args:
        image (numpy.ndarray): Original image in RGB format
        segmented_image (numpy.ndarray): Segmentation map in RGB format
        
    Returns:
        numpy.ndarray: Overlaid image
    """
    alpha = 0.5 # transparency for the original image
    beta = 1.0 # transparency for the segmentation map
    gamma = 0 # scalar added to each sum
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)
    return image

def save_plots(
    train_acc, valid_acc, 
    train_loss, valid_loss, 
    train_miou, valid_miou, 
    out_dir
):
    """
    Save training and validation metric plots.
    
    Args:
        train_acc (list): Training accuracy values
        valid_acc (list): Validation accuracy values
        train_loss (list): Training loss values
        valid_loss (list): Validation loss values
        train_miou (list): Training mIoU values
        valid_miou (list): Validation mIoU values
        out_dir (str): Directory to save plots
    """
    # Accuracy plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='tab:blue', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='tab:red', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'accuracy.png'))
    
    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='tab:blue', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='tab:red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'loss.png'))
    
    # mIOU plots.
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_miou, color='tab:blue', linestyle='-', 
        label='train mIoU'
    )
    plt.plot(
        valid_miou, color='tab:red', linestyle='-', 
        label='validataion mIoU'
    )
    plt.xlabel('Epochs')
    plt.ylabel('mIoU')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'miou.png'))
