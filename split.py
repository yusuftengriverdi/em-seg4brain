import os
import nibabel as nib
import numpy as np
from nilearn import image

def split_and_save_nii(input_file, output_dir, target_img):
    """
    Split a NIfTI file containing a stack of 4 images into separate NIfTI files.

    Parameters:
    - input_file: Path to the input NIfTI file.
    - output_dir: Directory to save the split NIfTI files.
    """

    # Load the NIfTI file
    nii_data = nib.load(input_file)
    nii_array = nii_data.get_fdata()

    # Assuming the input NIfTI file contains a stack of 4 images along the third dimension
    num_slices = nii_array.shape[-1]

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Split and save each slice as a separate NIfTI file
    for i in range(num_slices):
        # Extract the ith slice
        single_slice = nii_array[:, :, :, i]

        new_nii = nib.Nifti1Image(single_slice, affine=target_img.affine, header=target_img.header)

        # Create a new NIfTI image
        resampled = image.resample_img(new_nii, target_affine=target_img.affine, 
                                   target_shape=target_img.shape, 
                                   interpolation='continuous')

        # new_nii = nib.Nifti1Image(resampled, affine=target_img.affine, header=target_img.header)

        # Save the NIfTI image to the output directory
        output_file = os.path.join(output_dir, f"slice_{i+1}.nii")

        nib.save(resampled, output_file)

if __name__ == "__main__":
    # Example usage
    input_nii_file = "referenceSpace/atlas.nii"
    output_directory = "referenceSpace/"

    nii_data = nib.load("testingSet/testingStripped/1003.nii")
    split_and_save_nii(input_nii_file, output_directory, target_img=nii_data)
