import numpy as np

def compute_dice_coefficient(pred_mask, true_mask):
    # Convert inputs to boolean arrays
    pred_mask = pred_mask.astype(bool)
    true_mask = true_mask.astype(bool)
    
    # Compute intersection and union
    volume_intersect = np.logical_and(true_mask, pred_mask).sum()
    volume_true = true_mask.sum()
    volume_pred = pred_mask.sum()
    
    # Compute Dice coefficient
    dice_coefficient = 2. * volume_intersect / (volume_true + volume_pred)
    
    return dice_coefficient

# def compute_dice_coefficient(mask_pred, mask_gt):
#     """Compute soerensen-dice coefficient.

#     compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
#     and the predicted mask `mask_pred`.

#     Args:
#         mask_pred: 3-dim Numpy array of type bool. The predicted mask.
#         mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
      

#     Returns:
#         the dice coeffcient as float. If both masks are empty, the result is NaN
#     """
#     volume_sum = mask_gt.sum() + mask_pred.sum()
#     if volume_sum == 0:
#         return np.NaN
#     volume_intersect = (mask_gt & mask_pred).sum()
#     return 2 * volume_intersect / volume_sum


def compute_binary_iou(pred_mask, true_mask):
    cls = 1 # For background (0) and foreground (1)

    intersection = np.sum((pred_mask == cls) & (true_mask == cls))
    union = np.sum((pred_mask == cls) | (true_mask == cls))

    if union == 0:
        iou = float('nan')  # Handle cases where no union exists
    else:
        iou = intersection / union
    return iou

def compute_binary_recall(pred_mask, true_mask):
    # Ensure masks are binary (0 or 1)
    pred_mask = np.asarray(pred_mask).astype(np.bool_)
    true_mask = np.asarray(true_mask).astype(np.bool_)
    
    # True Positive (TP): correctly predicted positive cases
    TP = np.sum((pred_mask == 1) & (true_mask == 1))
    
    # False Negative (FN): actual positive cases predicted as negative
    FN = np.sum((pred_mask == 0) & (true_mask == 1))
    
    # Recall calculation
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    
    return recall


if __name__ == '__main__':
    pred_mask = np.array([[0, 1, 1], [0, 1, 0], [0, 0, 1]])
    true_mask = np.array([[0, 1, 0], [1, 1, 1], [0, 0, 1]])
    print('IoU:', compute_binary_iou(pred_mask, true_mask))
    print('Dice:', compute_dice_coefficient(pred_mask, true_mask))
    print('Recall:', compute_binary_recall(pred_mask, true_mask))