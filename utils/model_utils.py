import os
import torch
import torch.nn as nn
from PIL import Image

def predict(model, extractor, image, device):
    """
    Make a prediction on a single image.
    
    Args:
        model: The Segformer model
        extractor: The Segformer feature extractor
        image: The image in RGB format (PIL Image)
        device: The compute device
        
    Returns:
        torch.Tensor: The predicted class labels of shape [H, W]
    """
    pixel_values = extractor(image, return_tensors='pt').pixel_values.to(device)
    with torch.no_grad():
        logits = model(pixel_values).logits
    # Rescale logits to original image size.
    logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],  # PIL Image size is (W, H)
        mode='bilinear',
        align_corners=False
    )
    # Get class labels.
    labels = torch.argmax(logits.squeeze(), dim=0)
    return labels

class SaveBestModel:
    """
    Class to save the best model based on validation loss.
    """
    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss
        
    def __call__(self, current_valid_loss, epoch, model, out_dir, name='model'):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            model.save_pretrained(os.path.join(out_dir, name))

class SaveBestModelIOU:
    """
    Class to save the best model based on validation IoU.
    """
    def __init__(self, best_iou=float(0)):
        self.best_iou = best_iou
        
    def __call__(self, current_iou, epoch, model, out_dir, name='model'):
        if current_iou > self.best_iou:
            self.best_iou = current_iou
            print(f"\nBest validation IoU: {self.best_iou}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            model.save_pretrained(os.path.join(out_dir, name))

def save_model(model, out_dir, name='model'):
    """
    Function to save the trained model to disk.
    
    Args:
        model: The model to save
        out_dir (str): Directory to save the model
        name (str): Name for the model folder
    """
    model.save_pretrained(os.path.join(out_dir, name))
