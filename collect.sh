#!/bin/bash

# Source directory containing images
source_dir="registeredSet/mni/par0009/registeredImages"

# Destination directory
destination_dir="resultSet/mni/resultImages"

# Iterate over the images in the source directory
for image_path in "$source_dir"/*/result.1.nii; do
    # Check if the path exists and is a file
    if [ -e "$image_path" ] && [ -f "$image_path" ]; then
        # Extract the image name and map number
        image_name=$(basename "$(dirname "$image_path")")

        # Create the destination directory if it doesn't exist
        dest_dir="$destination_dir"
        mkdir -p "$dest_dir"

        # Copy the image to the destination directory with the new name
        cp "$image_path" "$dest_dir/$image_name.nii"

        echo "Image '$image_name' copied to '$dest_dir/$image_name.nii'"
    else
        echo "Error: Path '$image_path' does not exist or is not a file."
    fi
done


# Source directory containing images
source_dir="registeredSet/mni/par0009/registeredLabels"

# Destination directory
destination_dir="resultSet/mni/resultLabels"

# Declare an associative array to map tissue types to names
declare -A tissue_dict=(
    ["probabilisticMap1"]="CSF"
    ["probabilisticMap2"]="WM"
    ["probabilisticMap3"]="GM"
)

# Iterate over the images in the source directory
for image_path in "$source_dir"/*/*/result.nii; do
    # Check if the path exists and is a file
    if [ -e "$image_path" ] && [ -f "$image_path" ]; then
        # Extract the image name and tissue type
        image_name=$(basename "$(dirname "$(dirname "$image_path")")")
        tissue_type=$(basename "$(dirname "$image_path")")

        # Get the tissue name from the associative array
        tissue_name=${tissue_dict[$tissue_type]}

        # Create the destination directory if it doesn't exist
        dest_dir="$destination_dir"
        mkdir -p "$dest_dir"

        # Copy the image to the destination directory with the new name
        cp "$image_path" "$dest_dir/${image_name}_${tissue_name}.nii"

        echo "Image '$image_name' with tissue '$tissue_name' copied to '$dest_dir/${image_name}_${tissue_name}.nii'"
    else
        echo "Error: Path '$image_path' does not exist or is not a file."
    fi
done
