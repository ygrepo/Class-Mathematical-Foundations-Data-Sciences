import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy import signal
from matplotlib.colors import LogNorm
from plot_tools import *

"""
Given an image and a filter, returns the 2d circular convolution.
Arguments
img: 2d numpy array of the image of shape (R,C)
fil: 2d numpy array of the filter of shape (R,C)
Returns a 2d numpy array of shape (R,C) containing the circular convolution.  The 
returned value must be real (so return the real part if you do a complex calculation).
"""


def convolve(img, fil):
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.fft.fft2(fil)))


"""
Given a convolved image and a filter, returns the 2d circular deconvolution.
Arguments
img: 2d numpy array of the convolved image of shape (R,C)
fil: 2d numpy array of the filter of shape (R,C)
Returns a 2d numpy array of shape (R,C) containing the 2d circular deconvolution.  The 
returned value must be real (so return the real part if you do a complex calculation).
"""


def deconvolve(img, fil):
    eps = 1e-12
    return np.real(np.fft.ifft2(np.fft.fft2(img) / (np.fft.fft2(fil) + eps)))


"""
Given a convolved image, a filter, a list of images, and the variance
of the added Gaussian noise, returns the Wiener deconvolution.
Arguments
img: 2d numpy array of the convolved image of shape (R,C)
fil: 2d numpy array of the filter of shape (R,C)
imgs: 3d numpy array of n images from which to extract mean and variance info; shape (n,R,C)
v: the variance of each entry of the iid additive Gaussian noise
Returns a 2d numpy array of shape (R,C) containing the wiener deconvolution.  The 
returned value must be real (so return the real part if you do a complex calculation).
"""


def wiener_deconvolve(img, fil, imgs, v):
    n, r, c = imgs.shape
    var_dft_noise = r * c * v
    dft_imgs = np.array([np.fft.fft2(imgs[i, :, :]) for i in range(n)])
    mean_dft_imgs = np.mean(dft_imgs, axis=0)
    var_dft_imgs = np.var(dft_imgs, axis=0)
    dft_img = np.fft.fft2(img)
    dft_fil = np.fft.fft2(fil)
    dft_img -= dft_fil * mean_dft_imgs
    tmp = np.conj(dft_fil) * var_dft_imgs / (np.abs(dft_fil) ** 2 * var_dft_imgs + var_dft_noise)
    return np.real(np.fft.ifft2(mean_dft_imgs + tmp * dft_img))


def main():
    data = scipy.io.loadmat('olivettifaces.mat')['faces'].T.astype('float64')
    imgdb = np.array([im.reshape(64, 64).T for im in data])
    imgs = []
    imgs2 = []
    for i in range(6, 10):
        image = data[10 * i + 6, :].astype('float64')
        image = image.reshape(64, 64).T
        fil = np.outer(signal.gaussian(64, 1), signal.gaussian(64, 1))
        sfil = np.fft.ifftshift(fil)  # some filter
        conimg = convolve(image, sfil)
        deconimg = deconvolve(conimg, sfil)
        ran = np.amax(conimg) - np.amin(conimg)
        s = ran / 50.0
        noise = s * np.random.randn(*image.shape)
        noisycon = conimg + noise
        noisydecon = deconvolve(noisycon, sfil)
        wienerdecon = wiener_deconvolve(noisycon, sfil, imgdb, s ** 2)
        imgs.extend([image, fil, conimg, deconimg])
        imgs2.extend([noisycon, noisydecon, wienerdecon])

    coltitles = ['Image', 'Filter', 'Convolve', 'Deconvolve']
    coltitles2 = ['NoisyConvolve', 'NoisyDeconvolve', 'WienerDeconvolve']
    plot_image_grid(imgs, 'Noisefree', (64, 64), len(coltitles), 4, col_titles=coltitles)
    plot_image_grid(imgs2, 'Noisy', (64, 64), len(coltitles2), 4, col_titles=coltitles2)
    plot_fft_image_grid(imgs, 'NoisefreeFFT', (64, 64), len(coltitles), 4, col_titles=coltitles)


if __name__ == "__main__":
    main()
