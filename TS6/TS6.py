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
fs = 1000
zero_padd = 15

Rectangular = np.ones(N)
Bartlett    = sig.windows.bartlett(N)
Hann        = sig.windows.hann(N)
Blackman    = sig.windows.blackman(N)              
FlatTop     = sig.windows.flattop(N)

Rectangular = np.append(Rectangular,np.zeros(zero_padd*N))
Bartlett = np.append(Bartlett,np.zeros(zero_padd*N))
Hann = np.append(Hann,np.zeros(zero_padd*N))
Blackman = np.append(Blackman,np.zeros(zero_padd*N))
FlatTop = np.append(FlatTop,np.zeros(zero_padd*N))

fft_Rectangular = fft(Rectangular)/Rectangular.shape[0]
fft_Bartlett = fft(Bartlett)/Bartlett.shape[0]
fft_Hann = fft(Hann)/Hann.shape[0] 
fft_Blackman = fft(Blackman)/Blackman.shape[0] 
fft_FlatTop = fft(FlatTop)/Blackman.shape[0] 

ff = np.arange(0, fs, 1/(zero_padd+1))

ffr_Rectangular_log = 20*np.log10(np.abs(fft_Rectangular)/abs(fft_Rectangular[0])) 
fft_Bartlett_log = 20*np.log10(np.abs(fft_Bartlett)/abs(fft_Bartlett[0]))   
fft_Hann_log = 20*np.log10(np.abs(fft_Hann)/abs(fft_Hann[0]))   
fft_Blackman_log = 20*np.log10(np.abs(fft_Blackman)/abs(fft_Blackman[0]))   
fft_FlatTop_log = 20*np.log10(np.abs(fft_FlatTop)/abs(fft_FlatTop[0]))   

ff_aux = ff < 10

plt.plot(ff[ff_aux], ffr_Rectangular_log[ff_aux], label = "Rectangular")
plt.plot(ff[ff_aux], fft_Bartlett_log[ff_aux], label = "Bartlett")
plt.plot(ff[ff_aux], fft_Hann_log[ff_aux], label = "Hann")
plt.plot(ff[ff_aux], fft_Blackman_log[ff_aux], label = "Blackman")
plt.plot(ff[ff_aux], fft_FlatTop_log[ff_aux], label = "FlatTop")

plt.autoscale(enable = True, axis = 'x', tight = True)
plt.ylim([-150, 1])
plt.legend()
plt.grid()
