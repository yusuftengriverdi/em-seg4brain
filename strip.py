import os
import numpy as np
import nibabel as nib
from read import *
from tqdm import tqdm
def multiply_and_save_images(output_folder='testingSet/testingStripped'):
    """
    Multiply mask images with raw images and save them in the specified output folder.

    Args:
        output_folder (str, optional): Output folder to save the multiplied images.
    """
    os.makedirs(output_folder, exist_ok=True)

    masks, _ = readNIftI(which='Mask')  # Assuming masks are the mask images
    raw_images, tags = readNIftI(which='Images')  # Assuming Images are the raw images

    if masks is None or raw_images is None:
        print("Error reading images.")
        return

    for i, (mask, raw) in tqdm(enumerate(zip(masks, raw_images))):
        result_image = mask * raw
        result_image_nifti = nib.Nifti1Image(result_image, affine=np.eye(4))
        result_image_path = os.path.join(output_folder, f"{tags[i]}.nii")
        nib.save(result_image_nifti, result_image_path)
    
    print(f"Saved result for all!")


multiply_and_save_images()