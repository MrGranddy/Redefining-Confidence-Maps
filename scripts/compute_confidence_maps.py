import argparse
import os

import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Converts a MHA file to an UltraNerf dataset.")
    parser.add_argument("--params_path", type=str, help="The path to the parameters directory.")
    parser.add_argument("--output_path", type=str, help="The path to the output confidence maps.")
    args = parser.parse_args()

    # Load the parameters
    attenuation_total = np.load(os.path.join(args.params_path, "attenuation_total.npy"))
    reflection_total = np.load(os.path.join(args.params_path, "reflection_total.npy"))

    # Compute the confidence maps
    confidence_maps = attenuation_total * reflection_total

    # Save the confidence maps
    np.save(args.output_path, confidence_maps)