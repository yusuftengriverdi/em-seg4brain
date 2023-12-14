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
        normalized_data (numpy.ndarray): Input data to be scaled.
        intensity_min (int, optional): Minimum intensity value. Defaults to 0.
        intensity_max (int, optional): Maximum intensity value. Defaults to 255.

    Returns:
        numpy.ndarray: Scaled data.
    """
    # scaled_data = normalized_data / (intensity_max - intensity_min) + intensity_min
    scaled_data = np.array(cv2.normalize(data, None, intensity_min, intensity_max, cv2.NORM_MINMAX, cv2.CV_8U))
    return scaled_data.astype(np.int64)


def read_components(image_file, use_key = True):
    if image_file.endswith('.nii') or image_file.endswith('.nii.gz'):
        image_path = os.path.join(image_folder, image_file)

        # Load the image
        image_data = nib.load(image_path).get_fdata()
        image = image_data.astype(int)

        # Load the image
        if use_key: mask_file = image_file.replace('.nii', '_1C.nii')
        else: mask_file = image_file # Adjust this based on your label file naming
        mask_path = os.path.join(mask_folder, mask_file)
        mask_data = nib.load(mask_path).get_fdata()
        mask = mask_data.astype(int)

        # Load the corresponding label
        if use_key: label_file = image_file.replace('.nii', '_3C.nii') 
        else: label_file = image_file # Adjust this based on your label file naming
        label_path = os.path.join(label_folder, label_file)
        label_data = nib.load(label_path).get_fdata()
        labs = label_data

    return image, mask, labs

def calc_tissue_models(image_folder, label_folder, mask_folder):
    """
    Calculate tissue models.

    Args:
        image_folder (str): Path to the folder containing images.
        label_folder (str): Path to the folder containing labels.

    Returns:
        dict: Dictionary containing tissue models.
    """
    tissue_models = {}

    # List all files in the image folder
    image_files = os.listdir(image_folder)

    max_intensity = 256
    intensity_hist = np.zeros(max_intensity, dtype=int)

    for image_file in tqdm(image_files):
        
        image, mask, labs = read_components(image_file)
        
         # Exclude background (label 0)
        image = image * mask
        labs = labs * mask

        image = scale_to_intensity_range(image)

        # fig0 = multi_slice_viewer(image)
        # fig1 = multi_slice_viewer(labs)

        # plt.show()

        image, labs = image.flatten(), labs.flatten()
        
        # Extract CSF, WM, GM
        csf = image[labs == 1]
        wm = image[labs == 2]
        gm = image[labs == 3]

        # Calculate the histogram for the current image and add it to the accumulator
        hist = np.bincount(image, minlength=max_intensity)
        intensity_hist = intensity_hist + hist

        tissue_models[image_file] = {'CSF': csf, 'WM': wm, 'GM': gm}

    # Combine tissue samples
    t1 = np.concatenate([tissue_models[file]['CSF'] for file in tissue_models])
    t2 = np.concatenate([tissue_models[file]['WM'] for file in tissue_models])
    t3 = np.concatenate([tissue_models[file]['GM'] for file in tissue_models])

    # Avoid division by zero
    intensity_hist[intensity_hist == 0] = 1e10

    # Calculate tissue models
    tissue_model1 = (np.bincount(t1, minlength=max_intensity)) / intensity_hist
    tissue_model2 = (np.bincount(t2, minlength=max_intensity)) / intensity_hist
    tissue_model3 = (np.bincount(t3, minlength=max_intensity)) / intensity_hist

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


def tissue_model_based_segmentation(tissue_models, image_folder, label_folder, mask_folder, output_dir):
    csf, wm, gm = tissue_models

    # List all files in the image folder
    image_files = os.listdir(image_folder)

    max_intensity = 256

    for image_file in tqdm(image_files):
        image, mask, _ = read_components(image_file, use_key=False)

        image = scale_to_intensity_range(image)
        image = image * mask

        image_csf = image.copy()
        image_wm = image.copy()
        image_gm = image.copy()

        print(np.unique(csf, return_counts=True))
        for intensity in tqdm(range(0, 255)):
            image_csf[image_csf == intensity] = csf[intensity]
            image_wm[image_wm == intensity] = wm[intensity]
            image_gm[image_gm == intensity] = gm[intensity]

        fig0 = multi_slice_viewer(image_csf)
        fig1 = multi_slice_viewer(image_wm)
        fig2 = multi_slice_viewer(image_gm)

        plt.show()

    pass

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
                                label_folder= label_folder,
                                mask_folder=mask_folder,
                                output_dir='resultSet/tissue/resultLabels/')