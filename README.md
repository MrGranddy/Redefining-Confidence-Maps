# Ultrasound Confidence Maps with Neural Implicit Representation

This repository contains the implementation of the paper "Ultrasound Confidence Maps with Neural Implicit Representation" authored by V. Bugra Yesilkaynak, Vanessa Gonzalez Duque, Magdalena Wysocki, Yordanka Velikova, Diana Mateus, and Nassir Navab.

## Table of Contents

- [Ultrasound Confidence Maps with Neural Implicit Representation](#ultrasound-confidence-maps-with-neural-implicit-representation)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Methodology](#methodology)
    - [Ultra-NeRF Training](#ultra-nerf-training)
    - [Confidence Map Computation](#confidence-map-computation)
  - [Datasets](#datasets)
  - [Shadow Segmentation Using Random Forest](#shadow-segmentation-using-random-forest)
  - [References](#references)

## Introduction

The Ultrasound Confidence Map (CM) is an image representation used to indicate the reliability of pixel intensity values within ultrasound B-mode images. These maps correlate closely with the probability of sound reaching specific depths. Our methodology employs the physics-inspired intermediate representation maps from Ultra-NeRF[1] to compute CMs with an observation-angle awareness akin to clinical practice.

This repository includes code to replicate experiments and introduces a new dataset to facilitate further research into uncertainty quantification of B-mode images.

## Methodology

We compute CMs utilizing the physical principles of how echo traverses human tissue, employing the Ultra-NeRF technique. Ultra-NeRF receives the 3D absolute positions of the pixels in images and produces a rendered B-mode image that is then compared with the original image.

### Ultra-NeRF Training

Ultra-NeRF's training process utilizes its [official implementation](https://github.com/magdalena-wysocki/ultra-nerf). Post training, the "logs" directory will contain the intermediate representation maps which are essential for the next steps in our process.

To compute the confidence maps, use the `scripts/compute_confidence_maps.py` script with the `--params_path` set to the path containing the intermediate representation maps.

### Confidence Map Computation

To compute confidence maps, execute the `scripts/compute_confidence_maps.py` script with the following parameters:

- `--params_path`: Path to the intermediate representation maps.
- `--output_path`: Destination path for the computed confidence maps.

## Datasets

Our new dataset, `low limb leg`, along with two other datasets, `liver` and `spine`, are accessible. Download the `low limb leg` dataset from [here](https://drive.google.com/drive/folders/155NSsl98LdIYrrXGrYi5BJjogqDC08By?usp=sharing). Pre-calculated intermediate representation maps for this dataset are also available in the same location.

## Shadow Segmentation Using Random Forest

Below is a table presenting the quantitative results of shadow segmentation using various confidence map methods on the Liver dataset:

| Method | Dice Score | Hausdorff Distance | Precision |
|--------|------------|--------------------|-----------|
| RCM    | 0.4931     | 7.8495             | 0.6111    |
| MCM    | 0.4930     | 7.1471             | 0.6422    |
| ACM    | 0.4758     | 7.1475             | 0.6343    |
| **UCM**    | **0.5038**     | **5.5635**             | **0.7152** |

To conduct shadow segmentation experiments, execute the `shadow_segmentation/experiment.py` script. Necessary files for the liver experiment can be downloaded from the [Google Drive folder](https://drive.google.com/drive/folders/155NSsl98LdIYrrXGrYi5BJjogqDC08By?usp=sharing).

The script utilizes three files: `images.npy` (B-mode images), `masks.npy` (binary masks for bones or solid structures, whose underlay is considered as shadow), and `confidence_maps.npy` (for the chosen segmentation method).

## References

[1] M. Wysocki, M. F. Azampour, C. Eilers, B. Busam, M. Salehi, and N. Navab, "Ultra-NeRF: Neural Radiance Fields for Ultrasound Imaging," in *Medical Imaging with Deep Learning, MIDL 2023, Proceedings of Machine Learning Research*, vol. 227, pp. 382-401, PMLR, 2023.
