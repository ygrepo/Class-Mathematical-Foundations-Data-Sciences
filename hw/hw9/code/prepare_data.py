import numpy as np
import matplotlib.pyplot as plt
import scipy.io

def make_gaussian_filter(r,c,sigma=2) :
    f = lambda a,b : np.exp( -( (a-r/2.0)**2 + (b-c/2.0)**2 )/(2*sigma**2) )/(2 * np.pi)
    return np.fromfunction(f,(r,c))

def convolve(im, fil) :
    return np.fft.fftshift(np.fft.irfft2(np.fft.rfft2(im)*np.fft.rfft2(fil)))

def load_all_data(path):
    data = scipy.io.loadmat(path)['faces'].T.astype('float64')
    all_images = np.array([im.reshape(64,64).T for im in data])
    
    image = data[75,:].astype('float64')
    image = image.reshape(64,64).T
    filt = make_gaussian_filter(image.shape[0],image.shape[1],1)
    blur = convolve(image,filt)
    range = np.amax(blur) - np.amin(blur) 
    s = range/50
    noise = s*np.random.randn(*blur.shape)
    noisy_blur = blur + noise
    
    return all_images, image, filt, blur, noisy_blur, s