import random
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from zernpy import ZernPol, generate_phases_image


def get_zernike_image(coefficients: list, amplitude: tuple, image_width: int, image_height: int) -> np.ndarray:
    """
    Generate a Zernike image from a list of Zernike coefficients and amplitude
    Args:
        coefficients: list of tuples containing the Zernike coefficients
        amplitude: tuple containing the amplitude of the Zernike image
        image_width: width of the image
        image_height: height of the image
    Returns:
        zernike_image: Zernike image
    """
    # Validate coefficients
    if not isinstance(coefficients, list) or not all(isinstance(c, tuple) and len(c) == 2 and all(isinstance(i, int) for i in c) for c in coefficients):
        raise ValueError(
            'Coefficients must be a list of tuples, each containing two integers')
    if not isinstance(amplitude, tuple) or not all(isinstance(a, (int, float)) for a in amplitude):
        raise ValueError('Amplitude must be a tuple of integers or floats')
    polynomial = ()
    valid_amplitude = ()
    for i, (m, n) in enumerate(coefficients):
        # Skip vertical tilt, horizontal tilt and defocus Zernike polynomials
        if (m == -1 and n == 1) or (m == 1 and n == 1) or (m == 0 and n == 2):
            continue
        zp = ZernPol(m=m, n=n)
        polynomial += (zp,)
        valid_amplitude += (amplitude[i],)
    if len(polynomial) == 0:
        print('No valid Zernike coefficients were provided retiurning a blank image')
        zp = ZernPol(m=0,n=0)
        zernike_image = generate_phases_image((zp,), (0,), image_width, image_height)
        return zernike_image
    zernike_image = generate_phases_image(polynomial, valid_amplitude, image_width, image_height)
    return zernike_image


def get_phase_mask_aberration(phase_mask, coefficients, amplitude):
    shape = phase_mask.shape
    mask_disk = np.where(phase_mask != 0, 1, 0)
    indices = np.argwhere(mask_disk == 1)
    y1, x1 = indices.min(axis=0)
    y2, x2 = indices.max(axis=0)
    image_width = x2-x1
    image_height = y2-y1
    zernike_image = get_zernike_image(
        coefficients, amplitude, image_width, image_height)
    mask_abr = np.zeros(shape)
    mask_abr[y1:y2, x1:x2] = zernike_image
    # Normalize the mask to be between -1 and 1
    normalized_mask_abr = 2 * (mask_abr - np.min(mask_abr)) / (np.max(mask_abr) - np.min(mask_abr)) - 1
    #return normalized_mask_abr*mask_disk
    return mask_abr


def show_aberration_mask(phase_mask, coefficients, amplitude):
    mask_abr = get_phase_mask_aberration(phase_mask, coefficients, amplitude)
    fig, ax = plt.subplots(1, 3, figsize=(12, 8))
    im0 = ax[0].imshow(phase_mask, cmap='viridis')
    ax[0].set_title('Phase mask')
    cbar = fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
    im1 = ax[1].imshow(mask_abr, cmap='coolwarm')
    ax[1].set_title('Zernike mask')
    cbar = fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
    im2 = ax[2].imshow(mask_abr+phase_mask, cmap='viridis')
    ax[2].set_title('Zernike mask+Phase mask')
    cbar = fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

def generate_zernike_indices(max_n: int):
    """
    Generate all valid (n, m) index pairs for Zernike polynomials up to radial order max_n.
    Zernike polynomial conditions:
      1. n >= 0
      2. -n <= m <= n
      3. (n - |m|) is even
    Parameters
    ----------
    max_n : int
        The maximum radial order for n.
    Returns
    -------
    list of (int, int)
        A list of tuples (n, m) for all valid Zernike indices up to max_n.
    """
    return [
        (m,n)
        for n in range(max_n + 1)
        # Step by 2 ensures we only pick m values with the same parity as n
        for m in range(-n, n + 1, 2)
    ]

def random_zernike_parameters(max_n=5, max_num_abr=10):
    poly_pairs = generate_zernike_indices(max_n)
    num_abr = random.randint(1, max_num_abr)
    coefficients = []
    amplitude = []
    for i in range(num_abr):
        #TODO: Decide whether duplicates are allowed by usinf random.sample
        m, n = random.choice(poly_pairs)
        coefficients.append((m, n))
         # If both m and n are non-zero, pick a random amplitude; otherwise use 0
        if m != 0 and n != 0:
            amplitude.append(random.uniform(0, 1))
        else:
            amplitude.append(0)
    return coefficients, tuple(amplitude)

if __name__ == '__main__':
    mask_path = r'C:\Users\97254\Desktop\git\DeepSTORM3D\Mat_Files\mask_tetrapod_printed.mat'
    mask_dict = sio.loadmat(mask_path)
    mask_name = list(mask_dict.keys())[3]
    phase_mask = mask_dict[mask_name]
    for i in range(5):
        coefficients, amplitude = random_zernike_parameters()
        print(f'Coefficients: {coefficients}')
        print(f'Amplitude: {amplitude}')
        show_aberration_mask(phase_mask, coefficients, amplitude)
