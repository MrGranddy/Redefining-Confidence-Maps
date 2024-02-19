import numpy as np
import cv2

from skimage.exposure import match_histograms
from skimage.restoration import denoise_nl_means, estimate_sigma


import matplotlib.pyplot as plt

epsilon = 1e-6

def safe_sqrt(x):
    """Compute square root, setting any negative values to zero first."""
    return np.sqrt(np.maximum(x, 0))

def calculate_q(I):
    I_grad_x = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=3)
    I_grad_y = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(I_grad_x**2 + I_grad_y**2)
    laplacian = cv2.Laplacian(I, cv2.CV_64F)

    inside_sqrt = 0.5 * (grad_magnitude/(I + epsilon))**2 - 0.25 * (laplacian/(I + epsilon))**2
    q = safe_sqrt(inside_sqrt) / (1 + 0.25 * (laplacian/(I + epsilon))**2)
    
    return q

def estimate_q0(I, q):
    I_uint8 = (I*255).astype(np.uint8)  # Convert to 8-bit for thresholding
    _, binary_map = cv2.threshold(I_uint8, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    q_values_in_segment = q[binary_map == 255]
    q0 = np.median(q_values_in_segment)
    
    return q0

def calculate_c(q, q0, ccanny=1.0):
    c = ccanny / (1 + (q**2 - q0**2)/(q0**2 * (1 + q0**2)))
    return c

def apply_diffusion(I, c):
    diffused = I + c * cv2.Laplacian(I, cv2.CV_64F)
    return diffused

def anisotropic_diffusion(I, ccanny=1.0):
    q = calculate_q(I)
    q0 = estimate_q0(I, q)
    c = calculate_c(q, q0, ccanny)
    diffused = apply_diffusion(I, c)

    result = match_histograms(diffused, I)

    result = gaussian_filter(result, sigma=1.0)

    return result

def gaussian_filter(I, sigma=1.0):
    result = cv2.GaussianBlur(I, (7, 7), sigma)
    return result

def anisotropic_diffusion_2(img):
    sigma_est = np.mean(estimate_sigma(img, multichannel=False))
    patch_kw = dict(patch_size=5, patch_distance=6, multichannel=False)
    denoised = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=True, **patch_kw)
    return denoised


if __name__ == "__main__":

    import scipy.io

    I = scipy.io.loadmat("data/neck.mat")["img"]

    anisotropic = anisotropic_diffusion(I, ccanny=1.0)
    gaussian = gaussian_filter(I, sigma=1.0)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(I, cmap="gray")
    ax[0].set_title("Original")
    ax[1].imshow(anisotropic, cmap="gray")
    ax[1].set_title("Anisotropic")
    ax[2].imshow(gaussian, cmap="gray")
    ax[2].set_title("Gaussian")
    plt.show()
