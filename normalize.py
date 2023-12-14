import os
import numpy as np
import nibabel as nib

import os
import numpy as np
import nibabel as nib

from tqdm import tqdm

def scale_to_intensity_range(data, intensity_min=0, intensity_max=255):
    """
    Scale the data to the specified intensity range.

    Args:
        normalized_data (numpy.ndarray): Input data to be scaled.
        intensity_min (int, optional): Minimum intensity value. Defaults to 0.
        intensity_max (int, optional): Maximum intensity value. Defaults to 255.

    Returns:
        numpy.ndarray: Scaled data.
    """
    scaled_data = data * (intensity_max - intensity_min) + intensity_min
    return scaled_data
def scale_and_save_probabilistic_maps(input_folder, output_folder, intensity_min=0, intensity_max=255):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # List all files in the input folder
    file_list = os.listdir(input_folder)

    for file_name in tqdm(file_list):
        # Assuming the input files are in NIfTI format (modify if using a different format)
        if file_name.endswith('.nii') or file_name.endswith('.nii.gz'):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)

            # Load the normalized probabilistic map
            image = nib.load(input_path)
            data = image.get_fdata()

            # Scale to the intensity range [0, 255]
            scaled_data = scale_to_intensity_range(data)
            # Normalize to the range [0, 1]
            # min_value = np.min(data)
            # max_value = np.max(data)
            # normalized_data = (data - min_value) / (max_value - min_value)

            # Save the scaled probabilistic map
            scaled_image = nib.Nifti1Image(scaled_data, affine=image.affine)
            nib.save(scaled_image, output_path)

            print(f"Scaled to intensity range and saved: {output_path}")

# Example usage:
output_folder = "referenceSpace/norm/"
input_folder = "referenceSpace/"

scale_and_save_probabilistic_maps(input_folder, output_folder)