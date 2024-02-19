import argparse

import numpy as np

from utils import read_tracking, write_tracking

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Tracking Scaler')
    parser.add_argument('--input', type=str, help='path to input file', required=True)
    parser.add_argument('--output', type=str, help='path to output file', required=True)
    parser.add_argument('--scalex', type=float, default=1.0, help='scale factor for x')
    parser.add_argument('--scaley', type=float, default=1.0, help='scale factor for y')
    parser.add_argument('--scalez', type=float, default=1.0, help='scale factor for z')

    args = parser.parse_args()

    # Read tracking file
    tracking = read_tracking(args.input)

    # Create scaling matrix
    scaling = np.eye(4, dtype=np.float32)
    scaling[0, 0] = args.scalex
    scaling[1, 1] = args.scaley
    scaling[2, 2] = args.scalez

    # Apply scaling
    tracking = np.matmul(tracking, scaling)

    # Write tracking file
    write_tracking(args.output, tracking)

