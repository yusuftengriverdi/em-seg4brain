import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from viewer import multi_slice_viewer

def scale_to_intensity_range(data, intensity_min=0, intensity_max=255):
    """
    Scale the data to the specified intensity range.

    Args:
        data (numpy.ndarray): Input data to be scaled.
        intensity_min (int, optional): Minimum intensity value. Defaults to 0.
        intensity_max (int, optional): Maximum intensity value. Defaults to 255.

    Returns:
        numpy.ndarray: Scaled data.
    """
    scaled_data = np.array(cv2.normalize(data, None, intensity_min, intensity_max, cv2.NORM_MINMAX, cv2.CV_32F))
    return scaled_data

def read_components(image_file, use_key=True):
    """
    Read image components from NIfTI files.

    Args:
        image_file (str): File name of the image.
        use_key (bool, optional): Whether to use keys for file naming. Defaults to True.

    Returns:
        tuple: Tuple containing image, mask, and labels.
    """
    image_path = os.path.join(image_folder, image_file)

    # Load the image
    image = nib.load(image_path).get_fdata()

    # Load the mask
    mask_file = image_file.replace('.nii', '_1C.nii') if use_key else image_file
    mask_path = os.path.join(mask_folder, mask_file)
    mask_data = nib.load(mask_path).get_fdata()
    mask = mask_data.astype(int)

    # Load the labels
    label_file = image_file.replace('.nii', '_3C.nii') if use_key else image_file
    label_path = os.path.join(label_folder, label_file)
    label_data = nib.load(label_path).get_fdata()
    labels = label_data.astype(int)

    return image, mask, labels

def calc_tissue_models(image_folder, label_folder, mask_folder):
    """
    Calculate tissue models.

    Args:
        image_folder (str): Path to the folder containing images.
        label_folder (str): Path to the folder containing labels.

    Returns:
        tuple: Tuple containing dictionaries of CSF, WM, GM tissue models.
    """
    tissue_models = {}

    # List all files in the image folder
    image_files = os.listdir(image_folder)

    max_intensity = 256
    intensity_sum = np.zeros(max_intensity, dtype=int)

    for image_file in tqdm(image_files):
        image, mask, labels = read_components(image_file)

        # Normalize image.
        image = scale_to_intensity_range(image).astype(int)

        # Exclude background (label 0)
        image, labels = image[mask != 0], labels[mask != 0]
        
        # Extract CSF, WM, GM
        csf = image[labels == 1]
        wm = image[labels == 2]
        gm = image[labels == 3]

        # Calculate the histogram for the current image and add it to the accumulator
        hist = np.bincount(image.astype(int), minlength=max_intensity)
        intensity_sum = intensity_sum + hist

        tissue_models[image_file] = {'CSF': csf, 'WM': wm, 'GM': gm}

    # Combine tissue samples
    t1 = np.concatenate([tissue_models[file]['CSF'] for file in tissue_models])
    t2 = np.concatenate([tissue_models[file]['WM'] for file in tissue_models])
    t3 = np.concatenate([tissue_models[file]['GM'] for file in tissue_models])

    # Avoid division by zero
    intensity_hist = intensity_sum / len(image_files)
    
    plt.bar(range(len(intensity_hist)), height=intensity_hist)
    plt.show()

    # Calculate tissue models
    tissue_model1 = (np.bincount(t1.astype(int), minlength=max_intensity) / len(image_files)) / np.max(intensity_hist)
    tissue_model2 = (np.bincount(t2.astype(int), minlength=max_intensity) / len(image_files)) / np.max(intensity_hist)
    tissue_model3 = (np.bincount(t3.astype(int), minlength=max_intensity) / len(image_files)) / np.max(intensity_hist)
    
    # Plot the tissue models
    plt.plot(tissue_model1, label='CSF')
    plt.plot(tissue_model2, label='WM')
    plt.plot(tissue_model3, label='GM')
    plt.legend()
    plt.show()

    # Save tissue models to text files
    np.savetxt("tissueModel1.txt", tissue_model1)
    np.savetxt("tissueModel2.txt", tissue_model2)
    np.savetxt("tissueModel3.txt", tissue_model3)

    return tissue_model1, tissue_model2, tissue_model3

def tissue_model_based_segmentation(tissue_models, image_folder, output_dir):
    """
    Perform tissue model-based segmentation.

    Args:
        tissue_models (tuple): Tuple containing CSF, WM, GM tissue models.
        image_folder (str): Path to the folder containing images.
        output_dir (str): Path to the output directory.
    """
    csf, wm, gm = tissue_models

    # List all files in the image folder
    image_files = os.listdir(image_folder)

    for image_file in tqdm(image_files):
        image, mask, _ = read_components(image_file, use_key=False)

         # Exclude background (label 0)
        image = image * mask

        # Normalize image.
        image = scale_to_intensity_range(image).astype(int)

        image_csf = np.zeros_like(image).astype(csf.dtype)
        image_wm = np.zeros_like(image).astype(wm.dtype)
        image_gm = np.zeros_like(image).astype(gm.dtype)
        image_bg = (image == 0).astype(int)

        image = image.astype(int)
        for intensity in tqdm(range(0, 255)):
            image_csf[image == intensity] = csf[intensity]
            image_wm[image == intensity] = wm[intensity]
            image_gm[image == intensity] = gm[intensity]

        # Uncomment the following lines to visualize individual slices
        # fig = multi_slice_viewer(image)
        # fig0 = multi_slice_viewer(image_csf)
        # fig1 = multi_slice_viewer(image_wm)
        # fig2 = multi_slice_viewer(image_gm)

        # There's a mismatch in names, so I had to swap. 
        stacked = np.stack((image_bg, image_csf, image_gm, image_wm), axis=0)
        labels = np.argmax(stacked, axis=0)

        # Uncomment the following lines to visualize the segmented labels
        # fig3 = multi_slice_viewer(labels.astype(int))
        # plt.show()
        # print(np.unique(image_csf, return_counts=True))
        labels_nii = nib.Nifti1Image(labels, affine=np.eye(4), dtype=np.int8)
        image_csf_nii = nib.Nifti1Image(scale_to_intensity_range(image_csf, intensity_max=1), affine=np.eye(4))
        image_wm_nii = nib.Nifti1Image(scale_to_intensity_range(image_wm, intensity_max=1), affine=np.eye(4))
        image_gm_nii = nib.Nifti1Image(scale_to_intensity_range(image_gm, intensity_max=1), affine=np.eye(4))

        image_name = image_file.split(".nii")[0]
        nib.save(labels_nii, f'predictionSet/tissue_models/{image_name}.nii')
        nib.save(image_csf_nii, f'{output_dir}/{image_name}_CSF.nii')
        # There's a mismatch in names, so I had to swap. 
        nib.save(image_wm_nii, f'{output_dir}/{image_name}_GM.nii')
        nib.save(image_gm_nii, f'{output_dir}/{image_name}_WM.nii')

        pred_path = os.path.join('resultSet/custom/resultLabels/', f'{image_name}')
        pred_csf = nib.load(pred_path + '_CSF.nii').get_fdata() * mask
        pred_gm = nib.load(pred_path + '_GM.nii').get_fdata() * mask
        pred_wm = nib.load(pred_path + '_WM.nii').get_fdata() * mask
        # There's a mismatch in names, so I had to swap. 
        final_csf = scale_to_intensity_range(pred_csf + image_csf, intensity_max=1)
        final_wm = scale_to_intensity_range(pred_wm + image_gm, intensity_max=1)
        final_gm = scale_to_intensity_range(pred_gm + image_wm, intensity_max=1)

        final_stacked = np.stack((image_bg, final_csf, final_wm, final_gm), axis=0)
        final_labels = np.argmax(final_stacked, axis=0)

        # Uncomment the following lines to visualize the final segmented labels
        # fig1 = multi_slice_viewer(final_labels, "final")
        # plt.show()

        final_labels_nii = nib.Nifti1Image(final_labels, affine=np.eye(4), dtype=np.int8)
        final_csf_nii = nib.Nifti1Image(final_csf, affine=np.eye(4))
        final_wm_nii = nib.Nifti1Image(final_wm, affine=np.eye(4))
        final_gm_nii = nib.Nifti1Image(final_gm, affine=np.eye(4))

        nib.save(final_labels_nii, f'predictionSet/tissue_models_v_label_propagation/{image_name}.nii')
        nib.save(final_csf_nii, f'resultSet/custom_v_tissue/resultLabels/{image_name}_CSF.nii')
        nib.save(final_wm_nii, f'resultSet/custom_v_tissue/resultLabels/{image_name}_WM.nii')
        nib.save(final_gm_nii, f'resultSet/custom_v_tissue/resultLabels/{image_name}_GM.nii')

# Example usage:
image_folder = "trainingSet/trainingImages"
label_folder = "trainingSet/trainingLabels"
mask_folder = "trainingSet/trainingMask"

tissue_models = calc_tissue_models(image_folder, label_folder, mask_folder)

image_folder = "testingSet/testingImages"
label_folder = "testingSet/testingLabels"
mask_folder = "testingSet/testingMask"

tissue_model_based_segmentation(tissue_models=tissue_models,
                                image_folder=image_folder,
                                output_dir='resultSet/tissue/resultLabels/')
