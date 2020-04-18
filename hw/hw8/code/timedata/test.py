import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.fftpack as fft


fs = 100 # sample rate
rsample=50 # downsample frequency
fTwo=10 # frequency of the signal

n = np.arange(1024)
y = np.sin(2*np.pi*fTwo/fs*n)
y_res = signal.resample(y, len(n)/2)

Y = fft.fftshift(fft.fft(y))
f = -fs*np.arange(-512, 512)/1024
Y_res = fft.fftshift(fft.fft(y_res, 1024))
f_res = -fs/2*np.arange(-512, 512)/1024

plt.figure(1)
plt.subplot(211)
plt.stem(f, abs(Y))
plt.subplot(212)
plt.stem(f_res, abs(Y_res))
plt.show()