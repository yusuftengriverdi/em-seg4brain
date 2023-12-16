import viewer  # Assuming viewer is a module you have defined elsewhere
import numpy as np
import matplotlib.pyplot as plt
import os 
import imageio
from tqdm import tqdm 
import nibabel as nib
import SimpleITK as sitk
import cv2

def readNIftI(which='Labels', set='testing', directory=None):
    """
    Reads NIfTI files (Labels, Masks, or Images) and converts them to numpy arrays.

    Args:
        which (str, optional): Type of NIfTI file to read ('Labels', 'Mask', 'Images').

    Raises:
        NotImplementedError: If the specified type is not supported.
    """
    type_mapping = {
        'Labels': '_3C',
        'Mask': '_1C',
        'Images': ''
    }

    key = type_mapping.get(which)
    if key is None:
        raise NotImplementedError(f"'{which}' is not a supported type.")

    imgs = []
    tags = []
    if not directory: directory = f'{set}Set/{set}{which}/'
    for path in tqdm(os.listdir(directory)):
        tags.append(path.split(".nii")[0])
        path = os.path.join(directory, path)
        img = np.array(nib.nifti1.load(path).dataobj)
        imgs.append(img)
    return imgs, tags