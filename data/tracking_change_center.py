import argparse

import numpy as np

from utils import read_tracking, write_tracking

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Tracking Scaler')
    parser.add_argument('--input', type=str, help='path to input file', required=True)
    parser.add_argument('--output', type=str, help='path to output file', required=True)
    parser.add_argument('--height', type=int, default=512, help='height of the image')
    parser.add_argument('--width', type=int, default=512, help='width of the image')

    args = parser.parse_args()

    # Read tracking file
    tracking = read_tracking(args.input)

    # Create translation matrix
    translation = np.eye(4, dtype=np.float32)
    translation[1, 3] += args.height / 2 * 0.271318

    # Apply translation
    tracking = np.matmul(tracking, translation)

    # Write tracking file
    write_tracking(args.output, tracking)