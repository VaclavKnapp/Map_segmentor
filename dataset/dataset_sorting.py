import os
import shutil
import random

# Define paths
dataset_folder = '/OV-data/mapy/Crack_Segmentation_Dataset'
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
split_ratio = 0.8  # 80% for training, 20% for validation
split_index = int(len(image_files) * split_ratio)

train_image_files = image_files[:split_index]
train_mask_files = mask_files[:split_index]
valid_image_files = image_files[split_index:]
valid_mask_files = mask_files[split_index:]

# Move files to corresponding folders
for file in train_image_files:
    shutil.move(os.path.join(images_folder, file), os.path.join(train_images_folder, file))

for file in train_mask_files:
    shutil.move(os.path.join(masks_folder, file), os.path.join(train_masks_folder, file))

for file in valid_image_files:
    shutil.move(os.path.join(images_folder, file), os.path.join(valid_images_folder, file))

for file in valid_mask_files:
    shutil.move(os.path.join(masks_folder, file), os.path.join(valid_masks_folder, file))

print("Files have been successfully sorted into training and validation folders.")
