# -*- coding: utf-8 -*-
"""
Automatically detect rotation and line spacing of an image of text using
Radon transform

If image is rotated by the inverse of the output, the lines will be
horizontal (though they may be upside-down depending on the original image)

Note: It doesn't work with black borders
"""

import argparse
from skimage.transform import radon
from PIL import Image
import numpy as np
from numpy.fft import rfft
from numpy import argmax
import matplotlib.pyplot as plt

def rms_flat(a):
    return np.sqrt(np.mean(np.abs(a) ** 2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deskew images containing text using Radon Transform")
    parser.add_argument("-i", "--image", required=True, help="input image")
    args = parser.parse_args()

    # Load file, converting to grayscale
    image_file = args.image
    image = np.asarray(Image.open(image_file).convert("L"))
    image = image - np.mean(image)
    plt.subplot(2, 3, 1)
    plt.imshow(image)

    # Do the radon transform and show results
    sinogram = radon(image)
    plt.subplot(2, 3, 2)
    plt.imshow(sinogram.T, aspect="auto")
    plt.gray()

    # Find the RMS value of each row and find "busiest" rotation,
    # where the transform is lined up perfectly with the alternating dark
    # text and white lines
    r = np.array([rms_flat(line) for line in sinogram.transpose()])
    rotation = argmax(r)
    print('Rotation: {:.2f} degrees'.format(90 - rotation))
    plt.axhline(rotation, color='r')

    # Plot the busy row
    row = sinogram[:, rotation]
    N = len(row)
    plt.subplot(2, 3, 3)
    plt.plot(row)

    # Take spectrum of busy row and find line spacing
    window = np.blackman(N)
    spectrum = rfft(row * window)
    plt.plot(row * window)
    frequency = argmax(abs(spectrum))
    line_spacing = N / frequency  # pixels
    print('Line spacing: {:.2f} pixels'.format(line_spacing))

    plt.subplot(2, 3, 4)
    plt.plot(abs(spectrum))
    plt.axvline(frequency, color='r')
    plt.yscale('log')

    # Rotate original image and show
    image = Image.open(image_file).rotate(90 - rotation)
    plt.subplot(2, 3, 5)
    plt.imshow(image)
    plt.show()
