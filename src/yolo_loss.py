import math
import torch

def yolo_loss(y:torch.Tensor, y_pred:torch.Tensor, lambda_coord=5, lambda_noobj=0.5, num_classes=20, num_bounding_boxes=2):
    """
    The loss function is the one specified in the paper.
    Its result is the sum of differences in confidence scores, class probabilities 
    and localization coordinates and dimensions.

    ## Args

    - *y_pred*: tensor of model outputs. Each one follows the format [batch_size, S, S, (B * 5 + C)].
    - *y*: tensor matrix of shape (batch_size, grid_size, grid_size, num_of_bounding_boxes * 5 + C), 
                in which the 5 values are: confidence_value, x, y, w, h and the rest are the class probabilities.
                The class probabilities will be 0 for all classes except the target one, which will be 1.
                The confidence value will be 1 if there's an object. 0, otherwise. 
    """
    
    sum = 0

    # Add differences of x and y, as well as roots of w and h, for bounding boxes in cells with objects

    # FILTERING ----------------------------------------------------------------

    target_objs_indices = y[:, :, :, 0] == 1 # target_objs = target cells with confidence values == 1
    target_objs_bbs = y[target_objs_indices] # bounding boxes for each object (e.g. (2, 46))
    target_objs = target_objs_bbs[:, :5] # getting target bounding boxes (5 first values out of the 46 [confidence, x, y, w, h])
    target_objs_classes = target_objs_bbs[:, -num_classes:]

    predictions_with_objs = y_pred[target_objs_indices] # predictions for cells with objects

    class_predictions_with_obj = predictions_with_objs[:, -num_classes:]
    bb_predictions_with_obj = predictions_with_objs[:, :-num_classes]
    bb_predictions_with_obj = bb_predictions_with_obj.reshape(-1, num_bounding_boxes, 5) # reshaping (num_cells, B*5) to (num_cells, B, 5)


    max_confidences = torch.max(bb_predictions_with_obj[:, :, 4], 1, keepdim=True).values
    responsible_bb_predictions_with_obj = bb_predictions_with_obj[bb_predictions_with_obj[:, :, 4] == max_confidences]# get the predictions with the highest confidence. (10, 2, 5) -> (10, 5), since we now only have 1 bounding box per cell.

    # https://discuss.pytorch.org/t/how-to-negate-the-bool-type-tensor/51028/2
    predictions_noobj = y[~target_objs_indices] # predictions for cells other than the ones with objects
    predictions_noobj_bb = predictions_noobj[:, :-num_classes].reshape(-1, num_bounding_boxes, 5) # just the bounding boxes, reshaped to be of format (num_cells, num_bounding_boxes, 5)
    confidence_noobj_pred = predictions_noobj_bb[:, :, 4] # confidence values per bounding box (11, 2, 5) -> (11, 2) # all other values are excluded. Only one left is confidence (index 4).

    # CALCULATIONS -----------------------------------------------------------

    # Reminder:
    # -> format of predictions: (num_cells, 5), in which the 5 values are x, y, w, h, confidence
    # -> format of target: (num_cells, 5), in which the 5 values are confidence, x, y, w, h

    # For object ---
    x, x_pred = (target_objs[:, 1], responsible_bb_predictions_with_obj[:, 0])
    y, y_pred = (target_objs[:, 2], responsible_bb_predictions_with_obj[:, 1])
    w, w_pred = (target_objs[:, 3], responsible_bb_predictions_with_obj[:, 2])
    h, h_pred = (target_objs[:, 4], responsible_bb_predictions_with_obj[:, 3])
    c, c_pred = (target_objs[:, 0], responsible_bb_predictions_with_obj[:, 4])

    # Coordinates Comparison
    sum += lambda_coord * torch.sum((x - x_pred)**2 + (y - y_pred)**2)
    sum += lambda_coord * torch.sum((torch.sqrt(torch.abs(w))-torch.sqrt(torch.abs(w_pred)))**2 + (torch.sqrt(torch.abs(h))-torch.sqrt(torch.abs(h_pred)))**2 )

    # Confidence Comparison
    sum += torch.sum((c - c_pred)**2)

    # Class Comparison
    sum += torch.sum((target_objs_classes - class_predictions_with_obj)**2)

    # For no object ---
    
    # Confidence Comparison
    sum += lambda_noobj * torch.sum((0 - confidence_noobj_pred)**2) 
    
    #print(f'Shape of indices: {target_objs_indices.shape}, shape of target_objs: {target_objs.shape}, shape of predictions with objs: {predictions_with_objs.shape}')
    #print(target_objs.shape, class_predictions_with_obj.shape, bb_predictions_with_obj.shape)

    return sum