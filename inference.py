import os
import argparse
import cv2
import torch
from PIL import Image
from transformers import SegformerFeatureExtractor

import config
from model.model import segformer_model
from utils.model_utils import predict
from utils.visualization import draw_segmentation_map, image_overlay

def main(args):
    """
    Main inference function.
    
    Args:
        args: Command line arguments
    """
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = segformer_model(classes=config.ALL_CLASSES).to(device)
    
    # Load weights
    model_path = os.path.join(config.OUTPUT_DIR, args.model_name)
    if not os.path.exists(model_path):
        print(f"Model path {model_path} does not exist!")
        return
        
    model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")
    
    # Initialize feature extractor
    feature_extractor = SegformerFeatureExtractor(size=args.img_size)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process image or directory
    if os.path.isfile(args.input_path):
        process_image(args.input_path, model, feature_extractor, device, args.output_dir)
    elif os.path.isdir(args.input_path):
        for filename in os.listdir(args.input_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(args.input_path, filename)
                process_image(img_path, model, feature_extractor, device, args.output_dir)
    else:
        print(f"Input path {args.input_path} is neither a file nor a directory!")

def process_image(img_path, model, feature_extractor, device, output_dir):
    """
    Process a single image.
    
    Args:
        img_path (str): Path to the input image
        model: SegFormer model
        feature_extractor: Feature extractor for the model
        device: Device to use for inference
        output_dir (str): Directory to save output images
    """
    print(f"Processing {img_path}")
    
    # Load image
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    
    # Get prediction
    labels = predict(model, feature_extractor, pil_image, device)
    
    # Draw segmentation map
    segmentation_map = draw_segmentation_map(labels.cpu().numpy(), config.VIS_LABEL_MAP)
    
    # Overlay segmentation on original image
    result = image_overlay(image, segmentation_map)
    
    # Save results
    filename = os.path.basename(img_path)
    base_name, ext = os.path.splitext(filename)
    
    # Save segmentation map
    seg_map_path = os.path.join(output_dir, f"{base_name}_segmap{ext}")
    cv2.imwrite(seg_map_path, cv2.cvtColor(segmentation_map, cv2.COLOR_RGB2BGR))
    
    # Save overlaid image
    overlay_path = os.path.join(output_dir, f"{base_name}_overlay{ext}")
    cv2.imwrite(overlay_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    
    print(f"Results saved to {seg_map_path} and {overlay_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference with SegFormer for river segmentation')
    
    # Input and output paths
    parser.add_argument('--input_path', type=str, required=True,
                        help='path to input image or directory')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='directory to save results')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='model_iou',
                        help='name of the model directory (model_iou, model_loss, final_model)')
    parser.add_argument('--img_size', default=config.DEFAULT_IMAGE_SIZE, type=int, nargs='+',
                        help='image size [width, height]')
    
    args = parser.parse_args()
    
    main(args)
