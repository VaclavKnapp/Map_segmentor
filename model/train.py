import sys
import os
import argparse
import torch
from torch.optim.lr_scheduler import MultiStepLR
from transformers import SegformerFeatureExtractor

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from dataset.dataset import get_images, get_dataset, get_data_loaders
from model.model import segformer_model
from model.engine import train, validate
from utils.model_utils import save_model, SaveBestModel, SaveBestModelIOU
from utils.visualization import save_plots

def main(args):
    """
    Main training function.
    
    Args:
        args: Command line arguments
    """
    # Set random seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Create output directories
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.VALID_PREDS_DIR, exist_ok=True)

    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = segformer_model(classes=config.ALL_CLASSES).to(device)
    print(model)
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Get dataset paths
    train_images, train_masks, valid_images, valid_masks = get_images(args.dataset_path)
    
    # Print dataset size
    print(f"Training images: {len(train_images)}, Validation images: {len(valid_images)}")

    # Initialize feature extractor
    feature_extractor = SegformerFeatureExtractor(size=args.img_size)

    # Create datasets
    train_dataset, valid_dataset = get_dataset(
        train_images, 
        train_masks,
        valid_images,
        valid_masks,
        config.ALL_CLASSES,
        config.ALL_CLASSES,
        config.LABEL_COLORS_LIST,
        img_size=args.img_size,
        feature_extractor=feature_extractor
    )

    # Create data loaders
    train_dataloader, valid_dataloader = get_data_loaders(
        train_dataset, 
        valid_dataset,
        args.batch,
        args.num_workers
    )

    # Initialize model savers
    save_best_model = SaveBestModel()
    save_best_iou = SaveBestModelIOU()
    
    # Initialize LR scheduler
    scheduler = None
    if args.scheduler:
        scheduler = MultiStepLR(
            optimizer, milestones=[30, 60, 90], gamma=0.1, verbose=True
        )

    # Initialize metrics lists
    train_loss, train_pix_acc, train_miou = [], [], []
    valid_loss, valid_pix_acc, valid_miou = [], [], []
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"EPOCH: {epoch + 1}/{args.epochs}")
        
        # Train
        train_epoch_loss, train_epoch_pixacc, train_epoch_miou = train(
            model,
            train_dataloader,
            device,
            optimizer,
            config.ALL_CLASSES,
            alpha=args.alpha,
            beta=args.beta
        )
        
        # Validate
        valid_epoch_loss, valid_epoch_pixacc, valid_epoch_miou = validate(
            model,
            valid_dataloader,
            device,
            config.ALL_CLASSES,
            config.LABEL_COLORS_LIST,
            epoch,
            save_dir=config.VALID_PREDS_DIR,
            alpha=args.alpha,
            beta=args.beta,
            viz_map=config.VIS_LABEL_MAP
        )
        
        # Store metrics
        train_loss.append(train_epoch_loss)
        train_pix_acc.append(train_epoch_pixacc)
        train_miou.append(train_epoch_miou)
        valid_loss.append(valid_epoch_loss)
        valid_pix_acc.append(valid_epoch_pixacc)
        valid_miou.append(valid_epoch_miou)

        # Save best models
        save_best_model(
            valid_epoch_loss, epoch, model, config.OUTPUT_DIR, name='model_loss'
        )
        save_best_iou(
            valid_epoch_miou, epoch, model, config.OUTPUT_DIR, name='model_iou'
        )

        # Print metrics
        print(
            f"Train Epoch Loss: {train_epoch_loss:.4f}, "
            f"Train Epoch PixAcc: {train_epoch_pixacc:.4f}, "
            f"Train Epoch mIOU: {train_epoch_miou:.4f}"
        )
        print(
            f"Valid Epoch Loss: {valid_epoch_loss:.4f}, " 
            f"Valid Epoch PixAcc: {valid_epoch_pixacc:.4f}, "
            f"Valid Epoch mIOU: {valid_epoch_miou:.4f}"
        )
        
        # Update learning rate
        if scheduler:
            scheduler.step()
            
        print('-' * 50)

    # Save final model and plots
    save_model(model, config.OUTPUT_DIR, name='final_model')
    save_plots(
        train_pix_acc, valid_pix_acc, 
        train_loss, valid_loss,
        train_miou, valid_miou, 
        config.OUTPUT_DIR
    )
    print('TRAINING COMPLETE')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SegFormer for river segmentation')
    
    # Dataset and model parameters
    parser.add_argument('--dataset_path', type=str, default=config.DATASET_PATH,
                        help='path to the dataset')
    parser.add_argument('--img_size', default=config.DEFAULT_IMAGE_SIZE, type=int, nargs='+',
                        help='image size [width, height]')
    
    # Training parameters
    parser.add_argument('--epochs', default=config.DEFAULT_EPOCHS, type=int,
                        help='number of epochs to train for')
    parser.add_argument('--lr', default=config.DEFAULT_LEARNING_RATE, type=float,
                        help='learning rate for optimizer')
    parser.add_argument('--batch', default=config.DEFAULT_BATCH_SIZE, type=int,
                        help='batch size for data loader')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='number of workers for data loading')
    
    # Scheduler
    parser.add_argument('--scheduler', action='store_true',
                        help='use learning rate scheduler')
    
    # Loss function parameters
    parser.add_argument('--alpha', default=config.CHAMFER_WEIGHT, type=float,
                        help='weight for Chamfer distance in loss')
    parser.add_argument('--beta', default=config.IOU_WEIGHT, type=float,
                        help='weight for IoU in loss')
    
    # Other parameters
    parser.add_argument('--seed', default=42, type=int,
                        help='random seed for reproducibility')
    
    args = parser.parse_args()
    print(args)
    
    main(args)
