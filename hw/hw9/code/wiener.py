import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import prepare_data 

path = "wienerimages/"
fftmax = 1e4
fftmin = 1e-7


def plotfft(path,image) :
    global fftmin, fftmax
    fft = np.fft.fft2(image)
    ffts = np.abs(np.fft.fftshift(fft))
    freqs = np.fft.fftfreq(fft.shape[0])
    ex = (np.amin(freqs),np.amax(freqs))
    ax = plt.imshow(ffts,norm=LogNorm(vmin=fftmin,vmax=fftmax),extent=ex+ex)
    plt.colorbar(ax)
    plt.savefig(path +'.pdf',bbox_inches='tight')    
    plt.close()

def main() :
    global path
    if not os.path.exists(path):
        os.makedirs(path)

    all_images, filt, blur, noisy_blur, s = prepare_data.load_all_data('olivettifaces.mat')
    #all images is an np array consisting all 64x64 images in the dataset
    #filt is an np array with a gaussian blur filter 
    #blur is the result of convolving the true image with filt (4a)
    #noisy_blur is blur with added additive gaussian noise of standard deviation s (4b)
    # s is the standard deviation of noise added in noisy_blur above

    ax = plt.imshow(filt, cmap=plt.cm.gray)
    plt.colorbar(ax)
    plt.axis('off')
    plt.savefig(path + 'Filter'+'.pdf',bbox_inches='tight',norm=LogNorm())
    plt.close()
    plotfft(path+"FilterFFT",filt)


    plt.imshow(blur, cmap=plt.cm.gray)
    plt.axis('off')
    plt.savefig(path + 'BlurImage'+'.pdf',bbox_inches='tight')
    plt.close()
    plotfft(path+"BlurImageFFT",blur)


    plt.imshow(noisy_blur, cmap=plt.cm.gray)
    plt.axis('off')
    plt.savefig(path + 'NoisyImage'+'.pdf',bbox_inches='tight')
    plt.close()
    plotfft(path+"NoisyImageFFT",noisy_blur)

    

if __name__ == "__main__" :
    main()
