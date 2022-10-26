#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 21:18:57 2022

@author: nabcon
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

# def my_senoidal (N, freq_M, amplitud = 1, valor_medio = 0, freq = 1, fase = 0):
    
#     ts = 1/freq_M
    
#     tt = np.linspace(0, (N-1)*ts, N)
    
#     xx = amplitud * np.sin(2*np.pi*(freq)*tt + fase) + valor_medio
    
#     return(tt,xx)

fs = 1000 # frecuencia de muestreo
N = 1000   # cantidad de muestras
freq = fs/4
DC = 0
Amplitud = 2

Wbins = 10

realizaciones = 200
ts = 1/fs
tt = np.linspace(0, (N-1)*ts, N)

# funciones = np.arange(realizaciones*N).reshape(realizaciones, N)

noise = (np.random.rand(1,realizaciones) - 0.5) * 2 #ruido va desde -2 a 2

tt = np.linspace(0, (N-1)*ts, N).reshape((N,1))
Omega = (np.pi/2 + noise * ((np.pi*2/N)))*fs*tt


XX_sin = np.sin(Omega)*Amplitud

XX_fft_sin = np.fft.fft(XX_sin, axis = 0)/XX_sin.shape[0]

ff = np.arange(0, fs, fs/N)
bfrec = ff<= fs/2


#plt.plot(ff[ff <= fs/2],(2*np.abs(XX_fft_sin[ff <= fs/2 , :])**2))

estimacion_amp = np.abs(XX_fft_sin[250 , :])*2

Densidad_Potencia = 2*np.abs(XX_fft_sin)**2

sub_matriz = Densidad_Potencia[250-Wbins:250+Wbins+1, :]

Potencia_estimada = np.sum(sub_matriz, axis = 0)

Amplitud_estimada = np.sqrt(2*Potencia_estimada)

# vstack concatena verticalmente
Estimadores = np.vstack([estimacion_amp, Amplitud_estimada]).transpose()

# Histograma
fig, ax = plt.subplots(nrows=1 , ncols=1)
#fig.set_size_inches(16,12)
plt.figure(2)
ax.hist(Estimadores) 

Medianas = np.median(Estimadores, axis = 0)
Sesgo = np.median(Estimadores, axis = 0) - Amplitud

Varianza = np.mean((Estimadores - Medianas)**2, axis = 0)