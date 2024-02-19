import argparse

import nibabel as nib

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    data = nib.load(args.input).get_fdata()
    data = data * 255
    data = np.clip(data, 0, 255)
    data = data.astype(np.uint8)
    nib.save(nib.Nifti1Image(data, np.eye(4)), args.output)