# River Segmentation from Historical Maps using SegFormer with Combined Chamfer-IoU Loss

This repository contains code for river segmentation from historical maps using the SegFormer architecture with a combined Chamfer distance and IoU (Intersection over Union) loss function. The model is designed to segment rivers in historical maps, which is useful for geographic analysis, historical research, and understanding changes in river systems over time.

## Table of Contents
- [Introduction](#introduction)
- [Mathematical Foundation](#mathematical-foundation)
  - [Chamfer Distance](#1-chamfer-distance)
  - [IoU Loss](#2-iou-intersection-over-union)
  - [Combined Loss Function](#3-combined-loss-function-chamfer--iou)
  - [Final Mathematical Form](#4-final-mathematical-form)
- [Installation](#installation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Training](#training)
  - [Training Arguments](#training-arguments)
  - [Training Process](#training-process)
  - [Step-by-Step Training Guide](#step-by-step-training-guide)
- [Inference](#inference)
  - [Using the Inference Script](#using-the-inference-script)
  - [Programmatic Inference](#programmatic-inference)
- [Results and Evaluation](#results-and-evaluation)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)

## Introduction

River networks on historical maps provide valuable information about environmental changes, urban development, and geographic evolution over time. Automatic segmentation of these river networks enables researchers to quantify and analyze these changes efficiently.

This repository implements a deep learning approach for river segmentation using SegFormer, a state-of-the-art vision transformer architecture for semantic segmentation. A key innovation is the use of a combined Chamfer distance and IoU loss function, which is particularly well-suited for thin, winding structures like rivers in maps.

The thin, linear nature of rivers in historical maps makes them challenging to segment using standard approaches. Their appearance is similar to cracks in infrastructure, which has led to using dataset from crack detection being adapted for this purpose.

## Mathematical Foundation

Our approach uses a combined loss function that leverages both Chamfer distance and IoU (Intersection over Union). This section provides a detailed mathematical explanation of this loss function.

### 1. Chamfer Distance

Let $P = \{p_1, p_2, \ldots, p_m\}$ be the mask predicted by the model and $G = \{g_1, g_2, \ldots, g_n\}$ be the ground truth mask, where $p_i$ and $g_j$ represent individual points (pixel coordinates) in the predicted and ground truth masks respectively. 

The Chamfer distance between these two point sets is defined as:

$$\text{Chamfer}(P, G) = \sum_{p \in P} \min_{g \in G} \|p - g\|^2 + \sum_{g \in G} \min_{p \in P} \|g - p\|^2$$

This metric has two components:
1. The first term measures, for each point in the predicted mask, the squared distance to the nearest point in the ground truth mask
2. The second term measures, for each point in the ground truth mask, the squared distance to the nearest point in the predicted mask

The Chamfer distance effectively penalizes predictions that are far from the ground truth, making it particularly useful for thin structures where precise localization is important.

### 2. IoU (Intersection-over-Union)

IoU is typically defined over volumetric representations, segmentation masks, or bounding boxes:

$$\text{IoU}(S, \hat{S}) = \frac{|\hat{S} \cap S|}{|\hat{S} \cup S|}$$

where:
- $\hat{S}$ is the predicted segmentation mask
- $S$ is the ground truth mask

IoU measures the overlap between two regions. When using IoU as a loss function, we typically take $1 - \text{IoU}$ so that minimizing the loss corresponds to maximizing the IoU. This transforms the metric into a proper loss function.

### 3. Combined Loss Function: Chamfer + IoU

We combine these two loss functions by taking a weighted sum of the Chamfer distance and $1 - \text{IoU}$:

$$L = \alpha \cdot \text{Chamfer}(P, G) + \beta \cdot (1 - \text{IoU}(S, \hat{S}))$$

Where:
- $\text{Chamfer}(P, G)$ promotes proximity between predicted and ground truth point sets
- $1 - \text{IoU}(S, \hat{S})$ promotes maximizing overlap between predicted and ground truth regions
- $\alpha, \beta$ are hyperparameters that balance the two loss components

This combination provides several advantages:
- The Chamfer distance component helps with precise localization of thin structures
- The IoU component helps maintain good region-based segmentation
- The balance between point-based and region-based metrics leads to better overall segmentation

### 4. Final Mathematical Form

The final form of our combined loss function is:

$$L_{\text{combined}} = \alpha \sum_{p \in P} \min_{g \in G} \|p - g\|^2 + \alpha \sum_{g \in G} \min_{p \in P} \|g - p\|^2 + \beta (1 - \text{IoU}(S, \hat{S}))$$

This loss function has been implemented in `metrics/loss.py`, where the `ChamferIoULoss` class provides a PyTorch implementation of this mathematical formulation.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/river-segmentation.git
cd river-segmentation
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

Alternatively, you can install the package in development mode:
```bash
pip install -e .
```

### Requirements

- Python 3.7+
- PyTorch 1.9+
- Transformers 4.15+
- Albumentations 1.1+
- OpenCV 4.5+
- NumPy 1.20+
- Matplotlib 3.4+
- tqdm 4.62+
- Pillow 8.4+
- scikit-learn 1.0+

## Dataset

This project uses a DeepCrack segmentation Dataset from historical maps, where rivers in these maps resemble thin, winding structures (similar to cracks in appearance).

1. Download the dataset (preprocessed) from [Google Drive](https://drive.google.com/drive/folders/1mmIWtPfpbCBKSv1V0mytVGzuusuMKEwz?usp=share_link)

2. Extract the dataset to your preferred location.

3. The dataset should have the following structure (no need if you download from the Google link):
```
River_Segmentation_Dataset/
├── images/
│   ├── map1.jpg
│   ├── map2.jpg
│   └── ...
└── masks/
    ├── mask1.jpg
    ├── mask2.jpg
    └── ...
```

4. To split the dataset into training and validation sets, run the dataset_sorting.py script (no need if you download from the Google link):
```bash
python dataset/dataset_sorting.py --dataset_path /path/to/your/River_Segmentation_Dataset
```

After running this script, the dataset structure will be:
```
River_Segmentation_Dataset/
├── train_images/
├── train_masks/
├── valid_images/
└── valid_masks/
```

### Dataset Characteristics

- **Images**: Historical maps containing river networks
- **Masks**: Binary masks where rivers are represented in white (255,255,255) and background in black (0,0,0)
- **Resolution**: Variable, resized during training
- **Format**: JPEG or PNG images

## Project Structure

```
./
├── config.py              # Central configuration file
├── dataset/
│   ├── __init__.py        # Package initialization
│   ├── dataset_sorting.py # Script to split dataset
│   └── dataset.py         # Dataset loading and preprocessing
├── metrics/
│   ├── __init__.py        # Package initialization
│   ├── loss.py            # Loss function implementations
│   └── metrics.py         # Evaluation metrics
├── model/
│   ├── __init__.py        # Package initialization
│   ├── engine.py          # Training and validation loops
│   ├── model.py           # Model definition
│   └── train.py           # Main training script
├── utils/
│   ├── __init__.py        # Package initialization
│   ├── model_utils.py     # Model utility functions
│   ├── point_utils.py     # Point extraction utilities
│   └── visualization.py   # Visualization utilities
├── inference.py           # Script for inference on new images
├── requirements.txt       # Required packages
└── setup.py               # Package setup script
```

### Key Files and Their Purposes

- **config.py**: Central configuration file with dataset paths, model parameters, classes, and visualization settings
- **dataset/dataset_sorting.py**: Splits the dataset into training and validation sets
- **dataset/dataset.py**: Defines dataset classes, data loading, and augmentation
- **metrics/loss.py**: Implements the Chamfer-IoU loss function
- **metrics/metrics.py**: Evaluation metrics like IoU, Tversky, and Hausdorff distance
- **model/model.py**: Model architecture definition using SegFormer
- **model/engine.py**: Training and validation loops
- **model/train.py**: Main training script with argument parsing
- **utils/**: Various utility functions for visualization, model management, etc.
- **inference.py**: Script for running inference on new images

## Training

### Training Arguments

The training script accepts the following arguments:

- `--dataset_path`: Path to the dataset (default: from config.py)
- `--img_size`: Image size for training [width, height] (default: [448, 448])
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 0.001)
- `--batch`: Batch size (default: 8)
- `--num_workers`: Number of workers for data loading (default: 8)
- `--scheduler`: Use learning rate scheduler (default: False)
- `--alpha`: Weight for Chamfer distance in loss (default: 1.0)
- `--beta`: Weight for IoU in loss (default: 1.0)
- `--seed`: Random seed for reproducibility (default: 42)

### Training Process

The training process follows these steps:

1. **Dataset Preparation**: The dataset is split into training and validation sets
2. **Model Initialization**: SegFormer model is initialized with the 'nvidia/mit-b2' checkpoint
3. **Training Loop**: For each epoch:
   - Forward pass through the model
   - Calculate the combined Chamfer-IoU loss
   - Backpropagation and parameter update
   - Validation on the validation set
   - Save the best models based on validation loss and IoU
4. **Visualization**: Training and validation metrics are plotted and saved
5. **Model Saving**: The final model and the best models are saved

### Step-by-Step Training Guide

1. Set the dataset path in `config.py` or through an environment variable:
```python
# Set environment variable
export DATASET_PATH=/path/to/your/River_Segmentation_Dataset

# Or update in config.py
DATASET_PATH = '/path/to/your/River_Segmentation_Dataset'
```

2. Run the training script:
```bash
python model/train.py --epochs 100 --lr 0.001 --batch 8 --img_size 448 448 --alpha 1.0 --beta 1.0
```

You can adjust the weights of the Chamfer distance and IoU components in the loss function using the `--alpha` and `--beta` parameters.

3. Monitor the training progress:
   - Loss and metrics are printed for each epoch
   - Segmentation visualizations are saved in the `outputs/valid_preds/` directory
   - The best models are saved in the `outputs/` directory

4. After training, three models will be available:
   - `outputs/model_loss/`: Best model based on validation loss
   - `outputs/model_iou/`: Best model based on validation IoU
   - `outputs/final_model/`: Model after the final epoch

5. Plots of training and validation metrics are saved in the `outputs/` directory:
   - `accuracy.png`: Training and validation accuracy
   - `loss.png`: Training and validation loss
   - `miou.png`: Training and validation mean IoU

### Advanced Training Options

#### Learning Rate Scheduler

You can enable a learning rate scheduler with the `--scheduler` flag:
```bash
python model/train.py --scheduler
```

This uses a MultiStepLR scheduler with milestones at epochs 30, 60, and 90, reducing the learning rate by a factor of 0.1 at each milestone.

#### Hyperparameter Tuning

The most important hyperparameters to tune are:

- Learning rate (`--lr`): Controls the step size during optimization
- Batch size (`--batch`): Controls the number of samples processed at once
- Loss weights (`--alpha` and `--beta`): Control the balance between Chamfer distance and IoU in the loss function

For thin structures like rivers, increasing the weight of the Chamfer distance component (alpha) may improve results.

## Inference

There are two ways to perform inference with the trained model:

### Using the Inference Script

The `inference.py` script provides a convenient way to run inference on new images:

```bash
# Run inference on a single image
python inference.py --input_path path/to/your/image.jpg --output_dir ./results --model_name model_iou

# Run inference on all images in a directory
python inference.py --input_path path/to/your/images/folder --output_dir ./results --model_name model_iou
```

Arguments:
- `--input_path`: Path to input image or directory (required)
- `--output_dir`: Directory to save results (default: ./results)
- `--model_name`: Name of the model directory to use (default: model_iou)
- `--img_size`: Image size for inference [width, height] (default: from config.py)

The script will generate two output files for each input image:
1. `*_segmap.jpg`: The segmentation map showing river regions
2. `*_overlay.jpg`: The original image with the segmentation map overlaid

## Results and Evaluation

### Evaluation Metrics

The model is evaluated using several metrics:

1. **IoU (Intersection over Union)**: Measures the overlap between predicted and ground truth segmentation
2. **Pixel Accuracy**: Percentage of correctly classified pixels
3. **Visual Inspection**: Visualization of segmentation maps on validation images

### Interpreting Results

- **IoU**: Higher is better, with 1.0 being perfect overlap. For thin structures like rivers, IoU values above 0.5 are generally good.
- **Pixel Accuracy**: While important, this metric can be misleading for imbalanced classes (rivers typically occupy a small portion of the map).
- **Visual Inspection**: Often the most practical way to assess performance, especially for historical map analysis.

### Output Examples

During training, validation segmentation maps are saved in the `outputs/valid_preds/` directory. These show the model's predictions on validation samples, which can be used to qualitatively assess performance.

### Model Architecture

You can experiment with different SegFormer variants by modifying the checkpoint in `model/model.py`:

Available checkpoints include:
- 'nvidia/mit-b0' (smallest, fastest)
- 'nvidia/mit-b1'
- 'nvidia/mit-b2' (default)
- 'nvidia/mit-b3'
- 'nvidia/mit-b4'
- 'nvidia/mit-b5' (largest, most accurate)

### Data Augmentation

You can customize the data augmentation pipeline in `dataset/dataset.py`:

### Loss Function

You can experiment with the balance between Chamfer distance and IoU loss by adjusting the alpha and beta parameters:

```bash
# Emphasize Chamfer distance
python model/train.py --alpha 2.0 --beta 0.5

# Emphasize IoU
python model/train.py --alpha 0.5 --beta 2.0
```

## Troubleshooting

### GPU vs. CPU Training

The code automatically detects and uses a GPU if available:
```python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
```

Training on CPU will be significantly slower. For CPU-only training, consider:
- Reducing image size: `--img_size 256 256`
- Using a smaller model: Change to 'nvidia/mit-b0' in `config.py`
- Reducing batch size: `--batch 2`

### Logging and Debugging

To get more detailed logging, you can modify the progress bar in `model/engine.py`:

```python
prog_bar = tqdm(
    train_dataloader, 
    total=len(train_dataloader), 
    bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
    desc=f"Epoch {epoch+1}/{epochs}"  # Add epoch information
)
```

For debugging, you can add print statements to output intermediate values and shapes:

```python
print(f"Pixel values shape: {pixel_values.shape}")
print(f"Target shape: {target.shape}")
print(f"Logits shape: {logits.shape}")
```


## License

This project is licensed under the MIT License - see the LICENSE file for details.
