from dice import diceScore3dMask, diceScore3dMaskordered
import SimpleITK as sitk
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import nibabel as nib
import csv
from tqdm import tqdm
import viewer

def readCases(numCase, setting):
    # Reads ground truth and predicted segmentation for a given case
    gtPath = 'testingSet/testingLabels/' + str(numCase) + '.nii'
    lab = np.array(nib.nifti1.load(gtPath).get_fdata())

    filePath = f'predictionSet/{setting}/' + str(numCase) + '.nii'
    seg = np.array(nib.nifti1.load(filePath).get_fdata()).astype(np.int8)

    slice_id = 150

    # Visualization code (assuming it's correct)
    # plt.set_cmap('gray')
    # plt.subplot(2, 4, 1)
    # plt.imshow((seg == 0)[:, slice_id, :])
    # plt.title("Tissue 0 (Background)")
    # plt.axis("off")
    # plt.subplot(2, 4, 2)
    # plt.imshow((seg == 1)[:, slice_id, :])
    # plt.title("Tissue 1")
    # plt.axis("off")
    # plt.subplot(2, 4, 3)
    # plt.imshow((seg == 2)[:, slice_id, :])
    # plt.title("Tissue 2")
    # plt.axis("off")
    # plt.subplot(2, 4, 4)
    # plt.imshow((seg == 3)[:, slice_id, :])
    # plt.title("Tissue 3")
    # plt.axis("off")

    # plt.subplot(2, 4, 5)
    # plt.imshow((lab == 0)[:, slice_id, :])
    # plt.axis("off")
    # plt.subplot(2, 4, 6)
    # plt.imshow((lab == 1)[:, slice_id, :])
    # plt.axis("off")
    # plt.subplot(2, 4, 7)
    # plt.imshow((lab == 2)[:, slice_id, :])
    # plt.axis("off")
    # plt.subplot(2, 4, 8)
    # plt.imshow((lab == 3)[:, slice_id, :])
    # plt.axis("off")

    return seg, lab

def evaluateTissue(seg, mask):
    # Evaluates tissue using dice score
    return diceScore3dMaskordered(seg, mask)

if __name__ == '__main__':

    setting = 'atlas_into_em_label_propagation-custom'
    # Set up the CSV file for logging
    log_file_path = f"dice_score_{setting}.csv"

    with open(log_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write header row to the CSV file
        writer.writerow(["Image", "Tissue 1", "Tissue 2", "Tissue 3", "Average Score"])

        sum = np.array([0, 0, 0, 0])
        image_list = ["1003", "1004", "1005", "1018", "1019", "1023", "1024", "1025", "1038", 
                      "1039", "1101", "1104", "1107", "1110", "1113", "1113", "1116", "1119", 
                      "1122", "1125", "1128"
                      ]

        for i in tqdm(image_list):
            print(f"Processing image: {i}")
            seg, lab = readCases(i, setting=setting)
            score = evaluateTissue(seg, lab)
            sum = sum + np.array(score)
            print(score)

            # Write to the CSV file
            writer.writerow([i] + score)

            # Display the images (assuming you have a function to display them)
            plt.show()

        # Calculate and write the average score to the CSV file
        average_score = sum / len(image_list)
        writer.writerow(["Average"] + list(average_score))

    print(f"Average Dice Scores: {average_score}")
