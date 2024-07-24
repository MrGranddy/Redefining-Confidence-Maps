# Ultrasound Confidence Maps with Neural Implicit Representation

This repository contains the implementation of the paper "Ultrasound Confidence Maps with Neural Implicit Representation" by V. Bugra Yesilkaynak, Vanessa Gonzalez Duque, Magdalena Wysocki, Yordanka Velikova, Diana Mateus, and Nassir Navab.

## Table of Contents

1. [Introduction](#introduction)
2. [Methodology](#methodology)
3. [Datasets](#datasets)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Experiments and Results](#experiments-and-results)
7. [Discussion](#discussion)
8. [Conclusion](#conclusion)
9. [Acknowledgments](#acknowledgments)

## Introduction

Ultrasound Confidence Map (CM) is an image representation that indicates the reliability of the pixel intensity values presented within ultrasound B-mode images. These maps are highly correlated with the probability of sound reaching specific depths. Our method leverages the physics-inspired intermediate representation maps of Ultra-NeRF[1] to compute CMs with observation-angle awareness, similar to clinical practice.

This repository includes code to replicate experiments and the newly introduced dataset to facilitate further research in uncertainty quantification of B-mode images.

## Methodology

We propose to compute CMs using the physical principles of how echo traverses human tissue by utilizing Ultra-NeRF. The Ultra-NeRF method receives the 3D absolute positions of the pixels in the images and produces a rendered B-mode image that can be compared with the original image.

### Ultra-NeRF Training

This work depends on the Ultra-NeRF method to generate the intermediate representation maps. The training of Ultra-NeRF is done using the [official implementation](https://github.com/magdalena-wysocki/ultra-nerf) of the Ultra-NeRF method.

After training and rendering the dataset following the Ultra-NeRF method, in the "logs" directory, you will find the intermediate representation maps at `{DATASET_NAME}/output_maps_{DATASET_NAME}_model_{NUM_ITERATIONS}_0/params`

Then you can use the `scripts/compute_confidence_maps.py` script to compute the confidence maps with `--params_path` set to the path of the intermediate representation maps.

### Confidence Map Computation

Use the `scripts/compute_confidence_maps.py` script to compute the confidence maps. The script requires the following arguments:

- `--params_path`: Path to the intermediate representation maps.
- `--output_path`: Path to save the computed confidence maps.

## References

[1] M. Wysocki, M. F. Azampour, C. Eilers, B. Busam, M. Salehi, and N. Navab, "Ultra-NeRF: Neural Radiance Fields for Ultrasound Imaging," in *Medical Imaging with Deep Learning, MIDL 2023, Proceedings of Machine Learning Research*, vol. 227, pp. 382-401, PMLR, 2023.
