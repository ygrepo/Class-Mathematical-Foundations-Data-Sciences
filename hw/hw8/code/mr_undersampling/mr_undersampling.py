import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

plt.close('all')

factor = 20

def plot_img(knee_im, filename):
    n = knee_im.shape[0]

    ticks = np.arange(0, n, 100)
    tick_labels = ticks / factor

    plt.figure()
    ax = plt.gca()
    plt.imshow(knee_im, origin='lower', cmap='gray')
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)
    plt.ylabel(r"$t_1$ (cm)", fontsize=14)
    plt.xlabel(r"$t_2$ (cm)", fontsize=14)
    plt.tick_params(labelsize=13)
    plt.colorbar()
    plt.savefig(filename, bbox_inches="tight", cmap='gray')

def plot_kspace(knee_ft, filename, n=512):
    fc = 100
    f_tick_labels = np.arange(-3.0, 3.1, 1.0)
    f_ticks = fc + f_tick_labels / (factor / n)
    aux_diff = int((n - 2 * fc) / 2)
    knee_ft_aux = np.fft.fftshift(np.abs(knee_ft)) / n ** 2
    knee_ft_plot = knee_ft_aux[aux_diff:(n - aux_diff), aux_diff:(n - aux_diff)]
    plt.figure()
    ax = plt.gca()
    plt.imshow(knee_ft_plot, norm=colors.LogNorm(), origin='lower', cmap='gray')
    plt.colorbar()
    ax.set_xticks(f_ticks)
    ax.set_xticklabels(f_tick_labels)
    ax.set_yticks(f_ticks)
    ax.set_yticklabels(f_tick_labels)
    plt.ylabel(r'$k_1$ (1/cm)', fontsize=14)
    plt.xlabel(r'$k_2$ (1/cm)', fontsize=14)
    plt.tick_params(labelsize=13)
    plt.savefig(filename, bbox_inches="tight", cmap='gray')

knee_im_aux = 10000 * np.flipud(loadmat('image_knee.mat')['image_knee'].T)
knee_im = np.flipud(knee_im_aux)  # this is the pixel domain image of knee mri
print(knee_im.shape)
# sample plotting in the pixel domain
#plot_img(knee_im, "mri_samp_full.pdf")
#n = knee_im.shape[0]
#print(f"n={n}")


# take the fft of the pixel domain image
# work with this variable when undersampling
knee_ft = np.fft.fft2(knee_im)
#print(knee_ft.shape)


# plotting the fft
#plot_kspace(knee_ft, n, "mri_samp_full_fft.pdf")

# write your code to undersample knee_ft and use the plotting routines to generate plots in the pixel domain and fourier domain.

knee_ft_even_rows = np.zeros_like(knee_ft)
knee_ft_even_rows[1::2] = knee_ft[1::2]
plot_kspace(knee_ft_even_rows,  "mri_undersampling_even_rows_fft.pdf")
under_sampling_even_rows_img = np.fft.ifft2(knee_ft_even_rows)
plot_img(np.abs(under_sampling_even_rows_img), "mri_even_rows_recons.pdf")

knee_ft_even_columns = np.zeros_like(knee_ft)
knee_ft_even_columns[:,1::2] = knee_ft[:,1::2]
plot_kspace(knee_ft_even_columns, "mri_undersampling_even_columns_fft.pdf")
under_sampling_even_columns_img = np.fft.ifft2(knee_ft_even_columns)
plot_img(np.abs(under_sampling_even_columns_img), "mri_even_columns_recons.pdf")

knee_ft_even_rows_columns = np.zeros_like(knee_ft)
knee_ft_even_rows_columns[1::2] = knee_ft[1::2]
knee_ft_even_rows_columns[:,1::2] = knee_ft[:,1::2]
plot_kspace(knee_ft_even_rows_columns, "mri_undersampling_even_rows_columns_fft.pdf")
under_sampling_even_rows_columns_img = np.fft.ifft2(knee_ft_even_rows_columns)
plot_img(np.abs(under_sampling_even_rows_columns_img), "mri_even_rows_columns_recons.pdf")

