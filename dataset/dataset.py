import glob
import os
import sys
import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Add parent directory to path to import config and utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.point_utils import get_label_mask, set_class_values

def get_images(root_path=None):
    """
    Get paths to images and masks for training and validation.
    
    Args:
        root_path (str): Path to the dataset folder
        
    Returns:
        tuple: Lists of paths to train images, train masks, valid images, valid masks
    """
    if root_path is None:
        root_path = config.DATASET_PATH
        
    train_images = glob.glob(f"{root_path}/train_images/*")
    train_images.sort()
    train_masks = glob.glob(f"{root_path}/train_masks/*")
    train_masks.sort()
    valid_images = glob.glob(f"{root_path}/valid_images/*")
    valid_images.sort()
    valid_masks = glob.glob(f"{root_path}/valid_masks/*")
    valid_masks.sort()

    return train_images, train_masks, valid_images, valid_masks

def train_transforms(img_size):
    """
    Transforms/augmentations for training images and masks.

    Args:
        img_size (list): Image size [width, height]
        
    Returns:
        albumentations.Compose: Composition of transforms
    """
    train_image_transform = A.Compose([
        A.Resize(img_size[1], img_size[0], always_apply=True),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=25)
    ])
    return train_image_transform

def valid_transforms(img_size):
    """
    Transforms/augmentations for validation images and masks.

    Args:
        img_size (list): Image size [width, height]
        
    Returns:
        albumentations.Compose: Composition of transforms
    """
    valid_image_transform = A.Compose([
        A.Resize(img_size[1], img_size[0], always_apply=True),
    ])
    return valid_image_transform

class SegmentationDataset(Dataset):
    """
    Dataset class for semantic segmentation.
    
    Args:
        image_paths (list): List of paths to images
        mask_paths (list): List of paths to masks
        tfms (albumentations.Compose): Transforms to apply
        label_colors_list (list): List of RGB colors for each class
        classes_to_train (list): List of classes to train
        all_classes (list): List of all classes
        feature_extractor: Feature extractor for the model
    """
    def __init__(
        self, 
        image_paths, 
        mask_paths, 
        tfms, 
        label_colors_list,
        classes_to_train,
        all_classes,
        feature_extractor
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.tfms = tfms
        self.label_colors_list = label_colors_list
        self.all_classes = all_classes
        self.classes_to_train = classes_to_train
        self.class_values = set_class_values(
            self.all_classes, self.classes_to_train
        )
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float32')
        mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB).astype('float32')

        # Make all pixel > 0 as 255 (binarize the mask)
        im = mask > 0
        mask[im] = 255
        mask[np.logical_not(im)] = 0

        transformed = self.tfms(image=image, mask=mask)
        image = transformed['image'].astype('uint8')
        mask = transformed['mask']
        
        # Get 2D label mask
        mask = get_label_mask(mask, self.class_values, self.label_colors_list).astype('uint8')
        mask = Image.fromarray(mask)
               
        encoded_inputs = self.feature_extractor(
            Image.fromarray(image), 
            mask,
            return_tensors='pt'
        )
        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()

        return encoded_inputs
    
def get_dataset(
    train_image_paths, 
    train_mask_paths,
    valid_image_paths,
    valid_mask_paths,
    all_classes,
    classes_to_train,
    label_colors_list,
    img_size,
    feature_extractor
):
    """
    Create training and validation datasets.
    
    Args:
        train_image_paths (list): List of paths to training images
        train_mask_paths (list): List of paths to training masks
        valid_image_paths (list): List of paths to validation images
        valid_mask_paths (list): List of paths to validation masks
        all_classes (list): List of all classes
        classes_to_train (list): List of classes to train
        label_colors_list (list): List of RGB colors for each class
        img_size (list): Image size [width, height]
        feature_extractor: Feature extractor for the model
        
    Returns:
        tuple: Training and validation datasets
    """
    train_tfms = train_transforms(img_size)
    valid_tfms = valid_transforms(img_size)
    train_dataset = SegmentationDataset(
        train_image_paths,
        train_mask_paths,
        train_tfms,
        label_colors_list,
        classes_to_train,
        all_classes, 
        feature_extractor
    )
    valid_dataset = SegmentationDataset(
        valid_image_paths,
        valid_mask_paths,
        valid_tfms,
        label_colors_list,
        classes_to_train,
        all_classes,
        feature_extractor
    )
    return train_dataset, valid_dataset

def get_data_loaders(train_dataset, valid_dataset, batch_size, num_workers=8):
    """
    Create data loaders for training and validation.
    
    Args:
        train_dataset (Dataset): Training dataset
        valid_dataset (Dataset): Validation dataset
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: Training and validation data loaders
    """
    train_data_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        drop_last=False, 
        num_workers=num_workers,
        shuffle=True
    )
    valid_data_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        drop_last=False, 
        num_workers=num_workers,
        shuffle=False
    )
    return train_data_loader, valid_data_loader
