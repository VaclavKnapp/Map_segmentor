import sys
import os
from transformers import SegformerForSemanticSegmentation

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def segformer_model(classes, checkpoint=None):
    if checkpoint is None:
        checkpoint = config.MODEL_CHECKPOINT
        
    model = SegformerForSemanticSegmentation.from_pretrained(
        checkpoint,
        num_labels=len(classes),
    )
    return model
