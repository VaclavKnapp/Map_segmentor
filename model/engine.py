import torch
import torch.nn as nn
from tqdm import tqdm
from utils import draw_translucent_seg_maps
from metrics import IOUEval


def chamfer_distance(pred_points, target_points):
    """
    Compute the Chamfer distance between two point sets.
    """
    # Compute pairwise distance between predicted and target points
    dist_matrix = torch.cdist(pred_points.float(), target_points.float(), p=2)  # Shape: [N_p, N_t]

    # For each point in pred_points, find the min distance to target_points
    min_dist_p2t, _ = torch.min(dist_matrix, dim=1)  # Shape: [N_p]
    # For each point in target_points, find the min distance to pred_points
    min_dist_t2p, _ = torch.min(dist_matrix, dim=0)  # Shape: [N_t]

    # Sum the distances
    loss = (min_dist_p2t ** 2).mean() + (min_dist_t2p ** 2).mean()
    return loss


def train(
    model,
    train_dataloader,
    device,
    optimizer,
    classes_to_train
):
    print('Training')
    model.train()
    train_running_loss = 0.0
    prog_bar = tqdm(
        train_dataloader, 
        total=len(train_dataloader), 
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
    )
    counter = 0  # To keep track of batch counter
    num_classes = len(classes_to_train)
    iou_eval = IOUEval(num_classes)

    for i, data in enumerate(prog_bar):
        counter += 1
        pixel_values, target = data['pixel_values'].to(device), data['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values)

        logits = outputs.logits  # Shape: [batch_size, num_classes, H, W]
        upsampled_logits = nn.functional.interpolate(
            logits, size=target.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        predicted = upsampled_logits.argmax(dim=1)  # Shape: [batch_size, H, W]

        # Initialize batch loss
        batch_loss = 0.0

        # Loop over batch size
        for b in range(pixel_values.size(0)):
            pred_mask = predicted[b]  # Shape: [H, W]
            target_mask = target[b]   # Shape: [H, W]

            # Extract points for specified classes
            total_loss = 0.0
            for cls in classes_to_train:
                pred_points = (pred_mask == cls).nonzero(as_tuple=False)  # Shape: [N_p, 2]
                target_points = (target_mask == cls).nonzero(as_tuple=False)  # Shape: [N_t, 2]

                if pred_points.size(0) == 0 or target_points.size(0) == 0:
                    continue  # Skip if one of the point sets is empty

                # Compute Chamfer distance loss
                loss = chamfer_distance(pred_points, target_points)
                total_loss += loss

            batch_loss += total_loss

        # Average loss over batch size
        loss = batch_loss / pixel_values.size(0)
        train_running_loss += loss.item()

        # Backpropagation and parameter update
        loss.backward()
        optimizer.step()

        # Update IOU metrics
        iou_eval.addBatch(predicted.data, target.data)

    # Compute per-epoch loss and metrics
    train_loss = train_running_loss / counter
    overall_acc, per_class_acc, per_class_iou, mIOU = iou_eval.getMetric()
    return train_loss, overall_acc, mIOU


def validate(
    model,
    valid_dataloader,
    device,
    classes_to_train,
    label_colors_list,
    epoch,
    save_dir
):
    print('Validating')
    model.eval()
    valid_running_loss = 0.0
    num_classes = len(classes_to_train)
    iou_eval = IOUEval(num_classes)

    with torch.no_grad():
        prog_bar = tqdm(
            valid_dataloader, 
            total=(len(valid_dataloader)), 
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'
        )
        counter = 0 # To keep track of batch counter.
        for i, data in enumerate(prog_bar):
            counter += 1
            pixel_values, target = data['pixel_values'].to(device), data['labels'].to(device)
            outputs = model(pixel_values=pixel_values, labels=target)

            logits = outputs.logits
            upsampled_logits = nn.functional.interpolate(
                logits, size=target.shape[-2:], 
                mode="bilinear", 
                align_corners=False
            )
            
            # Save the validation segmentation maps.
            if i == 1:
                draw_translucent_seg_maps(
                    pixel_values, 
                    upsampled_logits, 
                    epoch, 
                    i, 
                    save_dir, 
                    label_colors_list,
                )

            ##### BATCH-WISE LOSS #####
            loss = outputs.loss
            valid_running_loss += loss.item()
            ###########################

            iou_eval.addBatch(upsampled_logits.max(1)[1].data, target.data)
        
    ##### PER EPOCH LOSS #####
    valid_loss = valid_running_loss / counter
    ##########################
    overall_acc, per_class_acc, per_class_iou, mIOU = iou_eval.getMetric()
    return valid_loss, overall_acc, mIOU