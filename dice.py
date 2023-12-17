import numpy as np
import cv2

def jaccardScore3dMask(seg, lab):
    """
    Computes the Jaccard score for 3D masks.

    Args:
    - seg (numpy.ndarray): Segmentation mask
    - lab (numpy.ndarray): Ground truth mask

    Returns:
    - float: Jaccard score
    """
    intersection, union = intersectionUnionCounter(seg, lab)
    if union != 0:
        return intersection / union
    else:
        return 1.0

def diceScore3dMask(seg, lab):
    """
    Computes the Dice score for 3D masks.

    Args:
    - seg (numpy.ndarray): Segmentation mask
    - lab (numpy.ndarray): Ground truth mask

    Returns:
    - list: List of Dice scores for each label
    """
    labels = [1, 2]
    
    max_dices = []
    for lab_label in labels:  
        max_dice = 0.0 
        max_label = 0
        for seg_label in labels:
            # Extract masks for the current labels
            seg_mask = (seg == seg_label).astype("uint8")
            lab_mask = (lab == lab_label).astype("uint8")
            
            # Calculate intersection and union
            intersection, union = intersectionUnionCounter(seg_mask, lab_mask)
            
            # Calculate Dice score
            dice = 2 * intersection / (union + intersection)
            
            if dice > max_dice:
                max_dice = dice
                max_label = seg_label

        print(lab_label, max_label)
        max_dices.append(max_dice)
    

    # Extract masks for the current labels
    seg_mask = (seg == 3).astype("uint8")
    lab_mask = (lab == 3).astype("uint8")
    # Calculate intersection and union
    intersection, union = intersectionUnionCounter(seg_mask, lab_mask)
    
    # Calculate Dice score
    dice = 2 * intersection / (union + intersection)
    
    max_dices.append(dice)


    max_dices.append(np.mean(max_dices))

    return max_dices


def diceScore3dMaskordered(seg, lab):
    """
    Computes the Dice score for 3D masks.

    Args:
    - seg (numpy.ndarray): Segmentation mask
    - lab (numpy.ndarray): Ground truth mask

    Returns:
    - list: List of Dice scores for each label and the average score
    """
    labels = [1, 2, 3]

    dices = []
    
    for label in labels:
        # Extract masks for the current label
        seg_mask = (seg == label).astype("uint8")
        lab_mask = (lab == label).astype("uint8")
        
        # Calculate intersection and union
        intersection, union = intersectionUnionCounter(seg_mask, lab_mask)
        
        # Calculate Dice score
        dice = 2 * intersection / (union + intersection)
        
        dices.append(dice)
    
    dices.append(np.mean(dices))

    return dices

@staticmethod
def intersectionUnionCounter(seg, lab):
    """
    Counts the intersection and union of two 3D masks.

    Args:
    - seg (numpy.ndarray): Segmentation mask
    - lab (numpy.ndarray): Ground truth mask

    Returns:
    - tuple: (intersection, union)
    """
    lab = lab.astype("uint8")
    seg = seg.astype("uint8")
    
    inter = 0
    union = 0
    
    for i in range(lab.shape[0]):
        inter += cv2.countNonZero(cv2.bitwise_and(lab[i], seg[i]))
        union += cv2.countNonZero(cv2.bitwise_or(lab[i], seg[i]))
    
    return inter, union
