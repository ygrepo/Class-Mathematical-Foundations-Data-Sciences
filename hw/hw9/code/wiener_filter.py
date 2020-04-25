import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from prepare_data import load_all_data
from scipy.signal import wiener as wiener_scipy

path = "wienerimages/"
fftmax = 1e4
fftmin = 1e-7


def plotfft(path, image):
    global fftmin, fftmax
    fft = np.fft.fft2(image)
    ffts = np.abs(np.fft.fftshift(fft))
    freqs = np.fft.fftfreq(fft.shape[0])
    ex = (np.amin(freqs), np.amax(freqs))
    ax = plt.imshow(ffts, norm=LogNorm(vmin=fftmin, vmax=fftmax), extent=ex + ex)
    plt.colorbar(ax)
    plt.savefig(path + '.pdf', bbox_inches='tight')
    plt.close()


"""
Given a convolved image and a filter, returns the 2d circular deconvolution.
Arguments
img: 2d numpy array of the convolved image of shape (R,C)
fil: 2d numpy array of the filter of shape (R,C)
Returns a 2d numpy array of shape (R,C) containing the 2d circular deconvolution.  The 
returned value must be real (so return the real part if you do a complex calculation).
"""


def deconvolve(img, fil, eps=1e-12):
    return np.fft.fftshift(np.fft.irfft2(np.fft.rfft2(img) / (np.fft.rfft2(fil) + eps)))


def denoise():
    all_images, image, filt, blur, noisy_blur, s = load_all_data('olivettifaces.mat')

    # original image
    plt.imshow(image, cmap=plt.cm.gray)
    plt.axis('off')
    plt.savefig(path + 'OriginalImage' + '.pdf', bbox_inches='tight')
    plt.close()
    plotfft(path + "OriginalImageFFT", image)

    # Deblurring img
    deblured_img = deconvolve(blur, filt)
    plt.imshow(deblured_img, cmap=plt.cm.gray)
    plt.axis('off')
    plt.savefig(path + 'DeBlurImage' + '.pdf', bbox_inches='tight')
    plt.close()
    plotfft(path + "DeBlurImageFFT", deblured_img)

    # Debluring noisy + blur
    deblured_img = deconvolve(noisy_blur, filt)
    plt.imshow(deblured_img, cmap=plt.cm.gray)
    plt.axis('off')
    plt.savefig(path + 'DeNoisyBlurImage' + '.pdf', bbox_inches='tight')
    plt.close()
    plotfft(path + "DeNoisyBlurImageFFT", deblured_img)


def wiener_2(imgs, img, fil, v):
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


def wiener(all_images, noisy_blur, filt, s):
    n, r, c = all_images.shape
    dft_imgs = np.array([np.fft.fft2(all_images[i, :, :]) for i in range(n)])
    dft_img = np.fft.fft2(noisy_blur)
    dft_fil = np.fft.fft2(filt)
    mean_dft_imgs = np.mean(dft_imgs, axis=0)
    var_dft_imgs = np.var(dft_imgs, axis=0)
    dft_img -= mean_dft_imgs
    var_dft_noise = (s ** 2) * r * c
    wiener_k = (np.conj(dft_fil) * var_dft_imgs) / ((np.abs(dft_fil) ** 2) * var_dft_imgs + var_dft_noise)
    return np.real(np.fft.ifft2(mean_dft_imgs + wiener_k * dft_img))


def wiener_deconvolve():
    all_images, image, filt, blur, noisy_blur, s = load_all_data('olivettifaces.mat')
    sfilt = np.fft.ifftshift(filt)
    deblured_img = wiener_2(all_images, noisy_blur, sfilt, s)
    plt.imshow(deblured_img, cmap=plt.cm.gray)
    plt.axis('off')
    plt.savefig(path + 'DeBlurNoisyImage' + '.pdf', bbox_inches='tight')
    plt.close()
    plotfft(path + "DeBlurNoisyImageFFT", deblured_img)



if __name__ == "__main__":
    #denoise()
    wiener_deconvolve()
