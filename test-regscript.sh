#!/bin/bash

# Record the start time
start_time=$(date +%s)

# Define the paths and parameters
elastix_fixed_input_dir="testingSet/testingStripped/"

transformix_input_dir_list=("referenceSpace/probabilisticMap1.nii" "referenceSpace/probabilisticMap2.nii" "referenceSpace/probabilisticMap3.nii")

elastix_moving_input_dir="referenceSpace/manualTemplate.nii"
param_affine="params/Par0009affine.txt"
param_elastic="params/Par0009elastic.txt"

elastix_output_dir="registeredSet/mni/par0009/registeredImages/"
transformix_label_output_dir="registeredSet/mni/par0009/registeredLabels/"


# List of image filenames to process
image_list=("1003.nii" "1004.nii" "1005.nii" "1018.nii" "1019.nii" "1023.nii" "1024.nii" "1025.nii" "1038.nii" "1039.nii" "1101.nii" "1104.nii" "1107.nii" "1110.nii" "1113.nii" "1113.nii" "1116.nii" "1119.nii" "1122.nii" "1125.nii" "1128.nii")

# Loop through the list and process each image
for image_file in "${image_list[@]}"; do
  # Create the output subdirectory for the current image
  image_name="${image_file%.nii}"
  elastix_output_subdir="$elastix_output_dir/$image_name/"

  # Create the 'elastixoutput' directory within the image's output directory
  if [ ! -d "$elastix_output_subdir" ]; then
    mkdir -p "$elastix_output_subdir"
  fi

  # Run the elastix command for the current image/patient.
  elastix -m "$elastix_moving_input_dir" -f "$elastix_fixed_input_dir/${image_name}.nii" -out "$elastix_output_subdir" -p "$param_affine" -p "$param_elastic"

  for map_file in "${transformix_input_dir_list[@]}"; do
    map_name=$(basename "$map_file" | sed 's/\..*//; s/probabilisticMap/probabilisticMap/')

    transformix_label_output_subdir="$transformix_label_output_dir/$image_name/$map_name/"

    echo $transformix_label_output_subdir
    # Create the 'transformixoutput' directory within the image's output directory
    if [ ! -d "$transformix_label_output_subdir" ]; then
      mkdir -p "$transformix_label_output_subdir"
    fi

    #!/bin/bash

    # Specify the path to your elastix parameter file
    parameter_file="$elastix_output_subdir/TransformParameters.1.txt"

    # Backup the original parameter file
    cp "$parameter_file" "${parameter_file}.bak"

    # Use awk to remove duplicate lines.
    awk '!/ResultImagePixelType/' "$parameter_file" > tmpfile && mv tmpfile "$parameter_file"
    awk '!/ResultImageFormat/' "$parameter_file" > tmpfile && mv tmpfile "$parameter_file"

    # Define the new parameters
    new_parameters="
    (ResultImageFormat \"nii\")
    (ResultImagePixelType \"float\")
    "
    # Append the new parameters to the parameter file
    echo "$new_parameters" >> "$parameter_file"

    # Display a message indicating the modification
    echo "Modified $parameter_file with new parameters."

    # Run the transformix command for the current image with the specified comment
    transformix -in "$map_file" -out "$transformix_label_output_subdir" -tp "$elastix_output_subdir/TransformParameters.1.txt"
  done
done

# Record the finish time
finish_time=$(date +%s)

# Calculate the total time in minutes
total_time=$(( (finish_time - start_time) / 60 ))

# Display the start and finish times
# echo "Script started at: $(date -d @$start_time '+%Y-%m-%d %H:%M:%S')"
# echo "Script finished at: $(date -d @$finish_time '+%Y-%m-%d %H:%M:%S')"
echo "Total time taken: $total_time minutes"
