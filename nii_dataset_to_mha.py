import os
import sys
import SimpleITK as sitk
from shutil import copy2

def convert_nii_to_mha(nii_path, mha_path):
    image = sitk.ReadImage(nii_path)
    sitk.WriteImage(image, mha_path)

def copy_and_convert(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        # Determine the path to the current directory in the output structure
        relative_path = os.path.relpath(root, input_dir)
        current_output_dir = os.path.join(output_dir, relative_path)
        
        # Create the directory in the output structure
        os.makedirs(current_output_dir, exist_ok=True)
        
        for file in files:
            input_file_path = os.path.join(root, file)
            
            # Convert .nii and .nii.gz files to .mha
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                output_file_name = os.path.splitext(file)[0] + '.mha'
                output_file_path = os.path.join(current_output_dir, output_file_name)
                convert_nii_to_mha(input_file_path, output_file_path)
            elif file.endswith('.csv'):
                # Copy .csv files directly
                output_file_path = os.path.join(current_output_dir, file)
                copy2(input_file_path, output_file_path)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_directory> <output_directory>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    copy_and_convert(input_dir, output_dir)
