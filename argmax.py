import os
import nibabel as nib
import numpy as np

def process_results(image_dir, label_dir, output_dir):
    """
    Process the saved result images.

    Args:
        image_dir (str): Directory containing result images.
        label_dir (str): Directory containing label images.
        output_dir (str): Directory to save the processed images.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over the images in the result directory
    for image_name in os.listdir(image_dir):
        if image_name.endswith('.nii'):
            image_path = os.path.join(image_dir, image_name)
            image_basename = os.path.splitext(image_name)[0]
            
            # Load the result image and label image
            result_image = nib.load(image_path).get_fdata()
            # Take the background from the result image
            background = result_image == 0

            mask = np.zeros_like(result_image)
            
            mask[~background] = 1
            labels = []

            for label_type in ["_CSF", "_WM", "_GM"]:
                label_path = os.path.join(label_dir, f"{image_basename}{label_type}.nii")

                # Check if the label file exists
                if not os.path.exists(label_path):
                    print(f"Error: Label file '{label_path}' not found.")
                    continue

                label_image = nib.load(label_path).get_fdata().astype(np.float32)
                labels.append(label_image * mask)

            # Binarize.
            mask[background] = 1
            mask[~background] = 0

            # Stack the background along with the specific label
            stacked_image = np.stack([mask, labels[0], labels[1], labels[2]], axis=-1)

            # Take np.argmax along the last axis
            processed_image = np.argmax(stacked_image, axis=-1).astype(np.float32)

            # Save the processed image
            output_path = os.path.join(output_dir, f"{image_basename}.nii")
            nib.save(nib.Nifti1Image(processed_image, affine=np.eye(4)), output_path)

            print(f"Processed image '{image_basename}' saved to '{output_path}'")

# Example usage
process_results("testingSet/testingStripped", "resultSet/mni/resultLabels", "predictionSet/label_propagation-mni")
