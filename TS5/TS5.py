#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 19:37:45 2022

@author: nabcon
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

def my_senoidal (N, freq_M, amplitud = 1, valor_medio = 0, freq = 1, fase = 0):
    
    ts = 1/freq_M
    
    tt = np.linspace(0, (N-1)*ts, N)
    
    xx = amplitud * np.sin(2*np.pi*(freq)*tt + fase) + valor_medio
    
    return(tt,xx)

fs = 1000 # frecuencia de muestreo
N = 1000   # cantidad de muestras
DC = 0
Amplitud = np.sqrt(2)

k0 = N/4
freq = k0*fs/N
freq_1 = (k0+0.75)*fs/N
freq_2 = (k0+0.25)*fs/N
freq_3 = (k0+0.5)*fs/N

zero_padd = 100


[t, Sr] = my_senoidal(N,fs,Amplitud, DC,freq)
[t, Sr_1] = my_senoidal(N,fs,Amplitud, DC,freq_1)
[t, Sr_2] = my_senoidal(N,fs,Amplitud, DC,freq_2)
[t, Sr_3] = my_senoidal(N,fs,Amplitud, DC,freq_3)

Sr = np.append(Sr,np.zeros(zero_padd*N))
Sr_1 = np.append(Sr_1,np.zeros(zero_padd*N))
Sr_2 = np.append(Sr_2,np.zeros(zero_padd*N))
Sr_3 = np.append(Sr_3,np.zeros(zero_padd*N))

Sr_fft = np.fft.fft(Sr)/Sr.shape[0]
Sr1_fft = np.fft.fft(Sr_1)/Sr_1.shape[0]
Sr2_fft = np.fft.fft(Sr_2)/Sr_2.shape[0]
Sr3_fft = np.fft.fft(Sr_3)/Sr_3.shape[0]

# ff = np.arange(0, fs, fs)
ff_2 = np.arange(0, fs, fs/((zero_padd+1)*N))
# ff_4 = np.arange(0, fs, fs/(4*N))
# ff_10 = np.arange(0, fs, fs/(10*N))

# bfrec = ff <= fs/2
bfrec = ff_2 <= fs/2

Area = np.sum(2*np.abs(Sr_fft[bfrec])**2)
Area_1 = np.sum(2*np.abs(Sr1_fft[bfrec])**2)
Area_2 = np.sum(2*np.abs(Sr2_fft[bfrec])**2)
Area_3 = np.sum(2*np.abs(Sr3_fft[bfrec])**2)

plt.clf()
plt.plot( ff_2[bfrec], (2*np.abs(Sr_fft[bfrec])**2),label = "freq = {:3.3f} Area {:3.3f}".format(freq,Area))
plt.plot( ff_2[bfrec], (2*np.abs(Sr1_fft[bfrec])**2),label = "freq = {:3.3f} Area {:3.3f}".format(freq_1,Area_1))
plt.plot( ff_2[bfrec], (2*np.abs(Sr2_fft[bfrec])**2),label = "freq = {:3.3f} Area {:3.3f}".format(freq_2,Area_2))
plt.plot( ff_2[bfrec], (2*np.abs(Sr3_fft[bfrec])**2),label = "freq = {:3.3f} Area {:3.3f}".format(freq_3,Area_3))
# plt.plot( ff_2[bfrec], 10*np.log10(2*np.abs(Sr_fft[bfrec])**2),label = "freq = {:3.3f} Area {:3.3f}".format(freq,Area))
# plt.plot( ff_2[bfrec], 10*np.log10(2*np.abs(Sr1_fft[bfrec])**2),label = "freq = {:3.3f} Area {:3.3f}".format(freq_1,Area_1))
# plt.plot( ff_2[bfrec], 10*np.log10(2*np.abs(Sr2_fft[bfrec])**2),label = "freq = {:3.3f} Area {:3.3f}".format(freq_2,Area_2))
# plt.plot( ff_2[bfrec], 10*np.log10(2*np.abs(Sr3_fft[bfrec])**2),label = "freq = {:3.3f} Area {:3.3f}".format(freq_3,Area_3))
plt.grid()
plt.xlim([245,255])
axes_hdl = plt.gca()
axes_hdl.legend()




