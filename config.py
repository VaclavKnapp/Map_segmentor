import os

# Dataset configuration
DATASET_PATH = os.environ.get('DATASET_PATH', './River_Segmentation_Dataset')
TRAIN_IMAGES_DIR = os.path.join(DATASET_PATH, 'train_images')
TRAIN_MASKS_DIR = os.path.join(DATASET_PATH, 'train_masks')
VALID_IMAGES_DIR = os.path.join(DATASET_PATH, 'valid_images')
VALID_MASKS_DIR = os.path.join(DATASET_PATH, 'valid_masks')

# Classes and colors
ALL_CLASSES = ['background', 'river']

LABEL_COLORS_LIST = [
    (0, 0, 0),  # Background
    (255, 255, 255),  # River
]

VIS_LABEL_MAP = [
    (0, 0, 0),  # Background
    (255, 255, 255),  # River
]

# Model configuration
MODEL_CHECKPOINT = 'nvidia/mit-b2'
OUTPUT_DIR = './outputs'
VALID_PREDS_DIR = os.path.join(OUTPUT_DIR, 'valid_preds')

# Training parameters
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 100
DEFAULT_IMAGE_SIZE = [448, 448]

# Loss function parameters
CHAMFER_WEIGHT = 1.0  # Alpha
IOU_WEIGHT = 1.0      # Beta
