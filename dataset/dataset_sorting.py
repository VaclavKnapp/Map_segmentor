import os
import shutil
import random
import argparse
import sys

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def sort_dataset(dataset_folder, split_ratio=0.8):
    """
    Sort dataset into training and validation sets.
    
    Args:
        dataset_folder (str): Path to the dataset folder
        split_ratio (float): Ratio of training to total samples
    """
    # Define paths
    images_folder = os.path.join(dataset_folder, 'images')
    masks_folder = os.path.join(dataset_folder, 'masks')

    # Create destination folders
    train_images_folder = os.path.join(dataset_folder, 'train_images')
    train_masks_folder = os.path.join(dataset_folder, 'train_masks')
    valid_images_folder = os.path.join(dataset_folder, 'valid_images')
    valid_masks_folder = os.path.join(dataset_folder, 'valid_masks')

    os.makedirs(train_images_folder, exist_ok=True)
    os.makedirs(train_masks_folder, exist_ok=True)
    os.makedirs(valid_images_folder, exist_ok=True)
    os.makedirs(valid_masks_folder, exist_ok=True)

    # Get list of files
    image_files = os.listdir(images_folder)
    mask_files = os.listdir(masks_folder)

    # Ensure matching filenames between images and masks
    image_files.sort()
    mask_files.sort()

    # Shuffle files
    combined = list(zip(image_files, mask_files))
    random.shuffle(combined)
    image_files[:], mask_files[:] = zip(*combined)

    # Split into training and validation sets
    split_index = int(len(image_files) * split_ratio)

    train_image_files = image_files[:split_index]
    train_mask_files = mask_files[:split_index]
    valid_image_files = image_files[split_index:]
    valid_mask_files = mask_files[split_index:]

    # Copy files to corresponding folders (using copy instead of move to preserve originals)
    for file in train_image_files:
        shutil.copy(os.path.join(images_folder, file), os.path.join(train_images_folder, file))

    for file in train_mask_files:
        shutil.copy(os.path.join(masks_folder, file), os.path.join(train_masks_folder, file))

    for file in valid_image_files:
        shutil.copy(os.path.join(images_folder, file), os.path.join(valid_images_folder, file))

    for file in valid_mask_files:
        shutil.copy(os.path.join(masks_folder, file), os.path.join(valid_masks_folder, file))

    print(f"Files have been successfully sorted into training and validation folders.")
    print(f"Training samples: {len(train_image_files)}, Validation samples: {len(valid_image_files)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sort dataset into training and validation sets')
    parser.add_argument('--dataset_path', type=str, default=config.DATASET_PATH,
                        help='Path to the dataset folder')
    parser.add_argument('--split_ratio', type=float, default=0.8,
                        help='Ratio of training to total samples')
    args = parser.parse_args()
    
    sort_dataset(args.dataset_path, args.split_ratio)
