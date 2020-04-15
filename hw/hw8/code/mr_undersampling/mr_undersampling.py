
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.colors as colors

plt.close('all')

knee_im_aux = 10000 * np.flipud(loadmat('image_knee.mat')['image_knee'].T)
knee_im = np.flipud(knee_im_aux) #this is the pixel domain image of knee mri

# sample plotting in the pixel domain

n = knee_im.shape[0]

factor = 20
ticks = np.arange(0,n,100)
tick_labels = ticks / factor

plt.figure()
ax = plt.gca()
im = plt.imshow(knee_im,origin='lower',cmap='gray')
ax.set_xticks(ticks)
ax.set_xticklabels(tick_labels)
ax.set_yticks(ticks)
ax.set_yticklabels(tick_labels)
plt.ylabel(r'$t_1$ (cm)',fontsize=14)
plt.xlabel(r'$t_2$ (cm)',fontsize=14)
plt.tick_params(labelsize=13)
plt.colorbar()
plt.savefig('mri_samp_full.pdf',bbox_inches="tight",cmap='gray')


#take the fft of the pixel domain image
#work with this variable when undersampling
knee_ft = np.fft.fft2(knee_im)


#plotting the fft
fc = 100
f_tick_labels = np.arange(-3.0,3.1,1.0)
f_ticks = fc + f_tick_labels / (factor/n)
aux_diff = int((n-2*fc)/2)
knee_ft_aux = np.fft.fftshift(np.abs(knee_ft))/ n**2
knee_ft_plot = knee_ft_aux[aux_diff:(n-aux_diff),aux_diff:(n-aux_diff)]
plt.figure()
ax = plt.gca()
im =plt.imshow(knee_ft_plot,norm=colors.LogNorm(),origin='lower',cmap='gray')
plt.colorbar()
ax.set_xticks(f_ticks)
ax.set_xticklabels(f_tick_labels)
ax.set_yticks(f_ticks)
ax.set_yticklabels(f_tick_labels)
plt.ylabel(r'$k_1$ (1/cm)',fontsize=14)
plt.xlabel(r'$k_2$ (1/cm)',fontsize=14)
plt.tick_params(labelsize=13)
plt.savefig('mri_samp_full_fft.pdf',bbox_inches="tight",cmap='gray')


#write your code to undersample knee_ft and use the plotting routines to generate plots in the pixel domain and fourier domain. 
