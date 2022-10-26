#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 21:59:52 2022

@author: nabcon
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.fft import fft, fftshift, fftfreq


N = 1000;
nn = np.arange(N)

zero_padd = 1

entrada = np.ones(N)
# hamming = 0.54 - 0.46*np.cos(2*np.pi*nn/(N-1))
# hanning = 0.5 - 0.5*np.cos(2*np.pi*nn/(N+1))
# blackman = 0.42 - 0.5*np.cos(2*np.pi*nn/(N-1)) + 0.08*np.cos(4*np.pi*nn/(N-1)) 

Bartlett    = sig.windows.bartlett(N)
hann        = sig.windows.hann(N)
Blackman    = sig.windows.blackman(N)              
FlatTop     = sig.windows.flattop(N)

Bartlett = np.append(Bartlett,np.zeros(zero_padd*N))

fft_Bartlett = fft(Bartlett)/Bartlett.shape[0]

ff = np.linspace(-0.5, 0.5, len(fft_Bartlett))
# t = np.arange(len(fft_Bartlett))
# ff = fftshift(fftfreq(t.shape[-1]))

plt.plot(ff, 20*np.log10(np.abs(fftshift(fft_Bartlett))))
#plt.plot(ff, 20*np.log10(np.abs(fftshift(fft_Bartlett/abs(fft_Bartlett).max()))))
plt.axis([-0.5, 0.5, -120, 0])
# fig, [ax1,ax2,ax3,ax4] = plt.subplots(nrows=4 , ncols=1)
# ax1.plot(Bartlett)
# ax2.plot(hann)
# ax3.plot(Blackman)
# ax4.plot(FlatTop)

 
#%% prub

from scipy.fft import fft, fftshift

window = sig.windows.bartlett(51)

plt.plot(window)

plt.title("Bartlett window")

plt.ylabel("Amplitude")

plt.xlabel("Sample")

plt.figure()

A = fft(window, 2048) / (len(window)/2.0)

freq = np.linspace(-0.5, 0.5, len(A))

response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))

plt.plot(freq, response)

plt.axis([-0.5, 0.5, -120, 0])

plt.title("Frequency response of the Bartlett window")

plt.ylabel("Normalized magnitude [dB]")

plt.xlabel("Normalized frequency [cycles per sample]")

