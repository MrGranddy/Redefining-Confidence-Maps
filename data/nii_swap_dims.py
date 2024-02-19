import argparse

import numpy as np

import nibabel as nib

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Nifti Swapper')
    parser.add_argument('--input', type=str, help='path to input file', required=True)
    parser.add_argument('--output', type=str, help='path to output file', required=True)
    args = parser.parse_args()

    # Read nifti file
    img = nib.load(args.input)
    data = img.get_fdata()

    # Swap dimensions
    data = np.swapaxes(data, 0, 1)
    data = np.swapaxes(data, 1, 2)
    data = np.swapaxes(data, 0, 1)
    data = np.swapaxes(data, 1, 2)

    # Write nifti file
    img = nib.Nifti1Image(data, img.affine, img.header)
    nib.save(img, args.output)