import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from tqdm import tqdm 
from viewer import *

import os
import numpy as np
import nibabel as nib
import cv2


def load_image(content_dir=..., modalities=['T1', 'T2_FLAIR'], case_number=1003, mask_dir=None):
    """
    Load image data from NIfTI files.

    Parameters:
    - content_dir: Directory containing image files.
    - modalities: List of modalities to load.
    - case_number: Case number for the specific image.
    - mask_dir: Directory containing mask files.

    Returns:
    - x: Transposed array of flattened image data.
    - volumes: List of loaded image volumes.
    - mask: Loaded mask volume.
    """
    nii_file_dir = os.path.join(content_dir, str(case_number))
    x = []
    volumes = []

    if modalities:
        for modality in modalities:
            nii_file_path = os.path.join(nii_file_dir, f"{modality}.nii")
            volume = nib.load(nii_file_path)
            data = volume.get_fdata()
            volumes.append(data)
            flattened = data.reshape(-1)
            x.append(flattened)
    else:
        nii_file_path = os.path.join(content_dir, str(case_number) + '.nii')
        volume = nib.load(nii_file_path)
        data = volume.get_fdata()

        nii_file_path = os.path.join(mask_dir, str(case_number) + '.nii')
        mask_volume = nib.load(nii_file_path)
        mask = mask_volume.get_fdata()

        # Ensure stripped.
        data *= mask

        volumes.append(data)
        flattened = data.reshape(-1)
        x.append(flattened)

    return np.asarray(x).transpose(1, 0), volumes, mask

def load_pred(case, key='label_propagation', atlas='custom', mask= None):
    """
    Load predictions for a specific case.

    Parameters:
    - case: Case identifier.
    - key: Key indicating the type of prediction ('label_prop' by default).
    - atlas: Atlas type ('custom' or 'mni') for probabilistic map loading.
    - mask: Binary mask indicating brain regions.

    Returns:
    - data: Loaded prediction data.
    """
    if key is None: key = atlas # In atlas mode.
    elif key == 'label_propagation': key = atlas # Get label propagation of custom.
    elif key == 'tissue_models': key = 'tissue' # Get tissue models as initialization.
    else: key = 'custom_v_tissue' # Get custom label propagations times tissues as initialization.


    nii_file_path = f'resultSet/{key}/resultLabels/{case}'
    
    print("SELECTED INIT/ATLAS MAP --> ", nii_file_path)

    pred_csf = nib.load(nii_file_path + '_CSF.nii').get_fdata()
    # There's a mismatch in names, so I had to swap. 
    pred_gm = nib.load(nii_file_path + '_WM.nii').get_fdata() 
    pred_wm = nib.load(nii_file_path + '_GM.nii').get_fdata()

    if mask is not None: 
        pred_csf *= mask
        pred_gm *= mask
        pred_wm *= mask

    else: print("Warning! Pred may not be skull stripped!")  

    # _ = multi_slice_viewer(pred_csf, title='init csf')  
    # _ = multi_slice_viewer(pred_gm, title='init gm')   
    # _ = multi_slice_viewer(pred_wm, title='init wm')   
    # plt.show()

    print("Succesfully loaded initial maps.")
    return [pred_csf, pred_gm, pred_wm]


def match_labels(a, mean, covariance):
    # Get the sorted indices of 'a'
    sorted_indices = np.argsort(a)
    # Use the sorted indices to reorder statistics.
    return a[sorted_indices], mean[sorted_indices], covariance[sorted_indices]

def random_init_params(x, K):
    """
    Initialize parameters of the model randomly.

    Parameters:
    - x: Input data.
    - K: Number of clusters.

    Returns:
    - a: Initial weights.
    - mean: Initial means.
    - covariance: Initial covariances.
    """
    a = np.ones(K) / K
    mean = [x[np.random.choice(x.shape[0])] for _ in range(K)]
    covariance = np.array([np.cov(x, rowvar=False) for _ in range(K)])

    return a, mean, covariance

def k_means_init_params(x, K, shape, orig=None):
    """
    Initialize parameters of the model using KMeans clustering.

    Parameters:
    - x: Input data.
    - K: Number of clusters.
    - shape: Shape of image volumes.

    Returns:
    - a: Initial weights.
    - mean: Initial means.
    - covariance: Initial covariances.
    """
    kmeans = KMeans(n_clusters=K, n_init=10, max_iter=100)
    label = kmeans.fit_predict(x)
    center = kmeans.cluster_centers_

    a = np.ones(K) / K
    for idx, clust in enumerate(range(K)):
        a[idx] = np.sum(label == clust) / label.shape[0]

    #     cluster_mask = (label == clust).reshape(shape)
    #     plt.subplot(1, K, idx + 1)
    #     plt.imshow(cluster_mask[:, :, cluster_mask.shape[2] // 2].T)
    #     plt.title(f'Cluster {clust}')
    #     plt.axis('off')

    # plt.tight_layout()
    # plt.show()
        
    mean = []
    covariance = []
    for clust in range(K):
        mean.append(center[clust])
        if orig is None:
            cov = np.cov(x[label == clust], rowvar=False)
        else:
            cov = np.cov(orig[label == clust], rowvar=False)
        
        covariance.append(cov + np.full_like(cov, fill_value=1e-6))

    return a, mean, covariance

def map_init_params(x, case, key, atlas, mask=None):
    """
    Initialize parameters for Gaussian Mixture Model (GMM) using label propagation segmentation.

    Parameters:
    - x: Input data.
    - K: Number of clusters for GMM.
    - case: Case number for the specific image.
    - key: Key indicating the type of prediction ('label_prop', etc.).
    - atlas: Atlas type ('custom' or 'mni') for probabilistic map loading.
    - mask: Binary mask indicating brain regions.

    Returns:
    - a: Initial weights for GMM components.
    - mean: Initial means for GMM components.
    - covariance: Initial covariances for GMM components.
    """

    # Load label propagation segmentation prediction
    prop_map = load_pred(case, key, atlas, mask)
    
    if mask is not None:
        # Extract indices of brain regions from the mask
        brain_mask = (mask != 0)
        # Take only non-zero indices for the algorithm and convert to 1-D.
        # Also rematch tissues by changing order, because there is a known mismatch.
        prop_map[0], prop_map[1],  prop_map[2] = prop_map[0][brain_mask],  prop_map[1][brain_mask], prop_map[2][brain_mask]

    else:
        raise ValueError("Mask should be provided in this implementation!")

    weight = np.array(prop_map).transpose(1, 0)

    return maximization(x, weight), weight


def log_likelihood(gmm_probs):
    """
    Calculate log likelihood of the model.

    Parameters:
    - gmm_probs: Gaussian mixture model probabilities.

    Returns:
    - Log likelihood.
    """
    return np.log(gmm_probs.sum(axis=1)).sum()

def expectation(a, mean, covariance, x):
    """
    Expectation step of the EM algorithm.

    Parameters:
    - a: Weights.
    - mean: Means.
    - covariance: Covariances.
    - x: Input data.

    Returns:
    - weight: Normalized weights of memberships.
    """

    k = len(mean)
    gmm_probs = np.zeros((x.shape[0], k))
    weight = np.zeros((x.shape[0], k))

    print("COV", covariance)
    print("MEAN", mean)
    print("ALPHA", a)

    for clust in range(k):
        gmm_probs[:, clust] = multivariate_normal.pdf(x, mean[clust], covariance[clust]) * a[clust] 

    for clust in range(k):
        weight[:, clust] = gmm_probs[:, clust] / (np.sum(gmm_probs, axis=1) + 1e-200)

    print("UPDATED WEIGHT", weight)
    return weight

def maximization(x, weight):
    """
    Maximization step of the EM algorithm.

    Parameters:
    - x: Input data.
    - weight: Normalized weights.

    Returns:
    - a: Updated weights.
    - mean: Updated means.
    - covariance: Updated covariances.
    """
    N, K = len(x), weight.shape[1]
    a = np.sum(weight, axis=0) / N
    mean = weight.T @ x / np.sum(weight, axis=0)[:, np.newaxis]
    covariance = []

    for k in range(K):
        diff = x - mean[k]
        weighted_diff = (weight[:, k][:, np.newaxis] * diff)
        covariance_ = diff.T @ weighted_diff / np.sum(weight[:, k]) + 1e-200
        covariance.append(covariance_)


    return a, mean, covariance

def e_m_algorithm(x, volumes, K, patience=5, init_mode='kmeans', max_iter=100, tol=1e-12, case_number=1, atlas = None, mask = None, atlas_mode = 'into'):
    """
    EM algorithm for clustering.

    Parameters:
    - x: Input data.
    - volumes: List of image volumes.
    - K: Number of clusters.
    - init_mode: Initialization mode ('kmeans', 'random', 'labprop', 'tissue').
    - max_iter: Maximum number of iterations.
    - tol: Convergence tolerance.
    - case_number: Case number for saving predictions.
    - use_atlas: Flag for using atlas.

    Returns:
    - weight: Final normalized weights.
    """
    print("Starting the algorithm...")
    N = len(x)
    weight = np.zeros(shape=(N, K))
    folder = f'atlas_{atlas_mode}_em_{init_mode}-{atlas}' if atlas_mode else f'em_{init_mode}'
    print("Selected folder: ", folder)

    # Ensure skull-stripping.
    brain_mask = (mask != 0) 
    for i, vol in enumerate(volumes):
        volumes[i] = vol * brain_mask

    x = np.expand_dims(volumes[0][brain_mask], axis = 1)
    if init_mode == 'kmeans':
        a, mean, covariance = k_means_init_params(x, K, shape=volumes[0].shape)
    elif init_mode == 'random':
        a, mean, covariance = random_init_params(x, K)
    else:
        (a, mean, covariance), _ = map_init_params(x, case_number, init_mode, atlas=atlas, mask = mask)

    if atlas is not None:
        _, atlas_weight = map_init_params(x, case_number, key=None, atlas=atlas, mask=mask)  

    prev_ll = 0
    counter = 0
    for _ in tqdm(range(max_iter)):
        weight = expectation(a, mean, covariance, x)
        print("MASK SHAPE: ", mask.shape)

        if atlas_mode == 'into':
            for clust in range(K):
                weight[:, clust] *= atlas_weight[:, clust]
                
        a, mean, covariance = maximization(x, weight)

        if init_mode == 'kmeans':
            a, mean, covariance = match_labels(np.array(a), np.array(mean), np.array(covariance))
        
        ll = log_likelihood(weight)

        print("ERROR: ", ll)

        if np.isclose(prev_ll, ll, atol=tol): counter += 1
        else: counter = 0

        if counter >= patience: break
        prev_ll = ll

    if atlas_mode == 'after':
        for clust in range(K):
            weight[:, clust] *= atlas_weight[:, clust]



    labels = (np.argmax(weight, axis=1)+1)

    if not mask is None:
        final_labels = mask.copy()
        background = (mask == 0)
        final_labels[brain_mask] = labels
        final_labels[final_labels == 0] = 1
        final_labels[background] = 0
        labels = final_labels
        pass
    
    fig = multi_slice_viewer(labels, title='final')
    plt.show()

    nii_img = nib.Nifti1Image(labels, affine=np.eye(4), dtype=np.int8)

    nib.save(nii_img, f'predictionSet/{folder}/{case_number}.nii')

    return weight

if __name__ == '__main__':

    import os

    content_dir = 'testingSet/testingStripped/'
    mask_dir = 'testingSet/testingMask/'
    case_number_list = os.listdir(content_dir)

    for i in tqdm(case_number_list):
        i = i.split(".nii")[0]
        x, volumes, mask_volume = load_image(content_dir=content_dir, modalities=[], case_number=i, mask_dir = mask_dir)
        e_m_algorithm(x, volumes, K=3, init_mode='kmeans', case_number=i, atlas=None, atlas_mode=None, mask=mask_volume)
