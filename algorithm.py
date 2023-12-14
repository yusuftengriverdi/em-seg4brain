import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from tqdm import tqdm 
from viewer import *

def load_image(content_dir='../p1/raw/', modalities=['T1', 'T2_FLAIR'], case_number=1, mask_dir = None):
    """
    Load image data from NIfTI files.

    Parameters:
    - content_dir: Directory containing image files.
    - modalities: List of modalities to load.
    - case_number: Case number for the specific image.

    Returns:
    - x: Transposed array of flattened image data.
    - volumes: List of loaded image volumes.
    """
    nii_file_dir = content_dir + str(case_number) + '/'
    x = []
    volumes = []

    if len(modalities) != 0:
        for modality in modalities:
            nii_file_path = nii_file_dir + modality + '.nii'
            volume = nib.load(nii_file_path)
            data = volume.get_fdata()
            volumes.append(data)
            flattened = data.reshape(-1)
            x.append(flattened)
    else:

        try:
            nii_file_path= content_dir + str(case_number) + '/' + 'result.1.img'
            volume = nib.load(nii_file_path)
            data = volume.get_fdata()
            volumes.append(data)
            flattened = data.reshape(-1)
            x.append(flattened)

        except Exception as e:

            nii_file_path= content_dir + '/' + str(case_number) + '.nii'
            volume = nib.load(nii_file_path)
            data = volume.get_fdata()


            nii_file_path= mask_dir + '/' + str(case_number) + '.nii'
            mask_volume = nib.load(nii_file_path)
            mask = mask_volume.get_fdata()
            data *= mask
            volumes.append(data)
            flattened = data.reshape(-1)
            x.append(flattened)

    return np.asarray(x).transpose(1, 0), volumes, mask

def load_prob_maps(atlas='mira'):

    volumes = []
    probs = []

    if atlas == 'mira':
        for i in range(1,4):
            nii_file_path = 'probabilisticMap' + str(i) + '.nii'
            volume = nib.load(nii_file_path)
            data = volume.get_fdata()
            volumes.append(data)
            flattened = data.reshape(-1)
            probs.append(data)
    
    
    if atlas == 'mni':
        for i in range(1,4):
            # nii_file_path = 'probabilisticMap' +i + '.nii'
            # volume = nib.load(nii_file_path)
            # data = volume.get_fdata()
            # volumes.append(data)
            # flattened = data.reshape(-1)
            # probs.append(flattened)
            pass
    
    return probs

def load_tissue_models():
    
    with open('tissueModel1.txt', 'r') as file:
        tissue1 = [float(line.strip()) for line in file.readlines()]
    with open('tissueModel2.txt', 'r') as file:
        tissue2 = [float(line.strip()) for line in file.readlines()]
    with open('tissueModel3.txt', 'r') as file:
        tissue3 = [float(line.strip()) for line in file.readlines()]

    return np.array([tissue1, tissue2, tissue3])

def load_pred(case, key='label_prop'):

    nii_file_path = 'preds/' + key + '/' + case + '.nii'
    volume = nib.load(nii_file_path)
    data = volume.get_fdata()
    # flattened = data.reshape(-1)    
    
    return data


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

def k_means_init_params(x, K, volumes, orig=None):
    """
    Initialize parameters of the model using KMeans clustering.

    Parameters:
    - x: Input data.
    - K: Number of clusters.
    - volumes: List of image volumes.

    Returns:
    - a: Initial weights.
    - mean: Initial means.
    - covariance: Initial covariances.
    """
    kmeans = KMeans(n_clusters=K)
    label = kmeans.fit_predict(x)
    center = kmeans.cluster_centers_

    a = np.ones(K) / K
    for idx, clust in enumerate(range(K)):
        a[idx] = np.sum(label == clust) / label.shape[0]

        cluster_mask = (label == clust).reshape(volumes[0].shape)
        plt.subplot(1, K, idx + 1)
        plt.imshow(cluster_mask[:, :, cluster_mask.shape[2] // 2].T)
        plt.title(f'Cluster {clust}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

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

def map_init_params(x, K, case, mode='tissue', tissue_maps=None, mask=None):
    """
    Initialize parameters using label propagation segmentation.

    Parameters:
    - x: Input data.
    - K: Number of clusters.
    - case: Case number for the specific image.
    - mode: Segmentation mode ('tissue', 'label_prop', etc.).
    - tissue_maps: List of tissue probability maps.

    Returns:
    - a: Initial weights.
    - mean: Initial means.
    - covariance: Initial covariances.
    """

    if tissue_maps is None:
        prop_map = load_pred(case=case, key=mode)
        
    mean = []
    covariance = []

    # Initialize weights using percentage of points in each cluster
    a = np.ones(K) / K
    figs = []
    
    if not mask is None:

        # Ensure that prop_map is skull-stripped.
        prop_map *= mask

        # Assign a brain mask to get indices.
        brain_mask = (mask != 0) 

        # Take only non-zero indices to the algorithm. 
        pred = prop_map[brain_mask]

        # print("Non-zero pred -->", pred.shape)

    for idx, clust in enumerate([1, 2, 3]):

        a[idx] = np.sum(prop_map == clust) / prop_map.size
        # Extract data points for the current label
        
        label_data = x[pred == clust]
        # figs.append(multi_slice_viewer(prop_map == clust, title=f'Cluster {clust}'))

        # Calculate mean and covariance for the current label
        label_mean = np.mean(label_data, axis=0)
        label_covariance = np.cov(label_data, rowvar=False)

        mean.append(label_mean)
        covariance.append(label_covariance)

    # plt.show()
    # Convert lists to arrays
    mean = np.array(mean)
    covariance = np.array(covariance)

    return a, mean, covariance


def log_likelihood(gmm_probs):
    """
    Calculate log likelihood of the model.

    Parameters:
    - gmm_probs: Gaussian mixture model probabilities.

    Returns:
    - Log likelihood.
    """
    return np.log(gmm_probs.sum(axis=1)).sum()

def expectation(a, mean, covariance, x, atlas = None, brain_mask = None):
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

    probabilistic_maps = None

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

def e_m_algorithm(x, volumes, K, init_mode='kmeans', max_iter=50, tol=1e-14, case_number=1, atlas = None, mask = None):
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
    # print("Starting the algorithm...")
    N = len(x)
    weight = np.zeros(shape=(N, K))
    folder = f'{init_mode}-em-atlas-{atlas}' if atlas else f'{init_mode}-em'
    # print(folder)

    if init_mode == 'kmeans':
        # print("kmeans!!!")
        a, mean, covariance = k_means_init_params(x, K, volumes)
    elif init_mode == 'random':
        # print("random!!!")
        a, mean, covariance = random_init_params(x, K)
    elif init_mode in ['label_prop', 'tissue'] :
        # print("label_prop or tissue!!!")
        brain_mask = (mask != 0) 
        volume = volumes[0]

        x = np.expand_dims(volume[brain_mask], axis = 1)
        # print("Non-zero x --> ", x.shape)
        a, mean, covariance = map_init_params(x, K, case=case_number, mode=init_mode, mask = mask)

    prev_ll = 0
    for _ in tqdm(range(max_iter)):
        weight = expectation(a, mean, covariance, x, atlas = atlas, brain_mask=brain_mask)
        if atlas:
            probabilistic_maps = load_prob_maps(atlas=atlas)

            for clust in range(K):
                weight[:, clust] *= probabilistic_maps[clust][brain_mask]
        
        a, mean, covariance = maximization(x, weight)
        ll = log_likelihood(weight)
        ll_diff = np.abs(ll - prev_ll)

        print("ERR: ", ll)
        if ll_diff < tol:
            break

        prev_ll = ll



    labels = (np.argmax(weight, axis=1))
    # print("before", np.unique(labels, return_counts=True))
    labels = (np.argmax(weight, axis=1)+1)
    # print("after", np.unique(labels, return_counts=True))

    if not mask is None:
        final_labels = mask.copy()
        background = (mask == 0)

        final_labels[brain_mask] = labels

        final_labels[final_labels == 0] = 1
        
        final_labels[background] = 0

        labels = final_labels
        pass
    
    # print("final", np.unique(labels, return_counts=True))
    fig = multi_slice_viewer(labels, title='final')
    plt.show()

    nii_img = nib.Nifti1Image(labels, affine=np.eye(4), dtype=np.int8)

    nib.save(nii_img, f'preds/{folder}/{case_number}.nii')

    return weight

if __name__ == '__main__':

    import os

    template = 'mira'

    if template == 'mira':
        content_dir = 'resultSet/param0009/resultImages_nii/'
        mask_dir = 'resultSet/param0009/resultMask_nii/'
        # content_dir = 'stripped/'
        case_number_list = os.listdir(content_dir)
    
    elif template == 'MNI':
        content_dir = 'resultSet(MNI)/param0009/resultImages_nii/'
        mask_dir = 'resultSet(MNI)/param0009/resultMask_nii/'
        # content_dir = 'stripped/'
        case_number_list = os.listdir(content_dir)


    for i in tqdm(case_number_list):
        i = i.split(".nii")[0]
        x, volumes, mask_volume = load_image(content_dir=content_dir, modalities=[], case_number=i, mask_dir = mask_dir)
        e_m_algorithm(x, volumes, K=3, init_mode='label_prop', case_number=i, atlas='mira', mask=mask_volume)
