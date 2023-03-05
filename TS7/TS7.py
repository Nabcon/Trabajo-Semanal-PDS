#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 21:18:57 2022

@author: nabcon
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig


fs = 1000 # frecuencia de muestreo
N = 1000   # cantidad de muestras
freq = fs/4
DC = 0
Amplitud = 2



realizaciones = 200
ts = 1/fs
tt = np.linspace(0, (N-1)*ts, N)

## reshape para no tener dimension 0
Bartlett    = sig.windows.bartlett(N).reshape(N,1)
Hann        = sig.windows.hann(N).reshape(N,1)
Blackman    = sig.windows.blackman(N).reshape(N,1)              
FlatTop     = sig.windows.flattop(N).reshape(N,1)

# funciones = np.arange(realizaciones*N).reshape(realizaciones, N)

noise = (np.random.rand(1,realizaciones) - 0.5) * 2 #ruido va desde -2 a 2

tt = np.linspace(0, (N-1)*ts, N).reshape((N,1))
Omega = (np.pi/2 + noise * ((np.pi*2/N)))*fs*tt

# aplico las ventanas a la senoidal
XX_sin_rect = np.sin(Omega)*Amplitud
XX_sin_Bartlett = XX_sin_rect*Bartlett
XX_sin_Hann = XX_sin_rect*Hann
XX_sin_Blackman = XX_sin_rect*Blackman
XX_sin_FlatTop = XX_sin_rect*FlatTop

# realizo la fft de las señales
XX_rect = np.fft.fft(XX_sin_rect, axis = 0)/XX_sin_rect.shape[0]
XX_Bart = np.fft.fft(XX_sin_Bartlett, axis = 0)/XX_sin_Bartlett.shape[0]
XX_Hann = np.fft.fft(XX_sin_Hann, axis = 0)/XX_sin_Hann.shape[0]
XX_Black = np.fft.fft(XX_sin_Blackman, axis = 0)/XX_sin_Blackman.shape[0]
XX_Flat = np.fft.fft(XX_sin_FlatTop, axis = 0)/XX_sin_FlatTop.shape[0]

ff = np.arange(0, fs, fs/N)
bfrec = ff<= fs/2
plt.figure(1)
plt.plot(ff[ff <= fs/2],10*np.log10(2*np.abs(XX_rect[ff <= fs/2 , :])**2))
plt.plot(ff[ff <= fs/2],(2*np.abs(XX_rect[ff <= fs/2 , :])**2))
plt.xlim([245,255])
plt.title("rectangular")
plt.ylim([-175, 1])

plt.figure(2)
plt.plot(ff[ff <= fs/2],10*np.log10(2*np.abs(XX_Bart[ff <= fs/2 , :])**2))
plt.plot(ff[ff <= fs/2],(2*np.abs(XX_Bart[ff <= fs/2 , :])**2))
plt.xlim([245,255])
plt.ylim([-175, 1])
plt.title("Bartlett")

plt.figure(3)
plt.plot(ff[ff <= fs/2],10*np.log10(2*np.abs(XX_Hann[ff <= fs/2 , :])**2))
plt.plot(ff[ff <= fs/2],(2*np.abs(XX_Hann[ff <= fs/2 , :])**2))
plt.xlim([245,255])
plt.ylim([-175, 1])
plt.title("Hann")

plt.figure(4)
plt.plot(ff[ff <= fs/2],10*np.log10(2*np.abs(XX_Black[ff <= fs/2 , :])**2))
plt.plot(ff[ff <= fs/2],(2*np.abs(XX_Black[ff <= fs/2 , :])**2))
plt.xlim([245,255])
plt.ylim([-175, 1])
plt.title("Blackman")

plt.figure(5)
plt.plot(ff[ff <= fs/2],10*np.log10(2*np.abs(XX_Flat[ff <= fs/2 , :])**2))
plt.plot(ff[ff <= fs/2],(2*np.abs(XX_Flat[ff <= fs/2 , :])**2))
plt.xlim([245,255])
plt.ylim([-175, 1])
plt.title("FlatTop")

#%% estimadores
from pandas import DataFrame
from IPython.display import HTML
Wbins = 3
#rectangular
Amplitud_Slice_Rect = 2*np.abs(XX_rect[250 , :]) # nos quedamos con una feta y obtemos la amplitud
Densidad_Potencia_rect = 2*np.abs(XX_rect)**2
sub_matriz_rect = Densidad_Potencia_rect[250-Wbins:250+Wbins, :]
Potencia_estimada_rect = np.sum(sub_matriz_rect, axis = 0)
Amplitud_Integral_Rect = np.sqrt(2*Potencia_estimada_rect)

#Bartlett
Amplitud_Slice_Bart = 2*np.abs(XX_Bart[250 , :])
Densidad_Potencia_Bart = 2*np.abs(XX_Bart)**2
sub_matriz_Bart = Densidad_Potencia_Bart[250-Wbins:250+Wbins, :]
Potencia_estimada_Bart = np.sum(sub_matriz_Bart, axis = 0)
Amplitud_Integral_Bart = np.sqrt(2*Potencia_estimada_Bart)

#Hann
Amplitud_Slice_Hann = 2*np.abs(XX_Hann[250 , :])
Densidad_Potencia_Hann = 2*np.abs(XX_Hann)**2
sub_matriz_Hann = Densidad_Potencia_Hann[250-Wbins:250+Wbins, :]
Potencia_estimada_Hann = np.sum(sub_matriz_Hann, axis = 0)
Amplitud_Integral_Hann = np.sqrt(2*Potencia_estimada_Hann)

#BlackMan
Amplitud_Slice_Black = 2*np.abs(XX_Black[250 , :])
Densidad_Potencia_Black = 2*np.abs(XX_Black)**2
sub_matriz_Black = Densidad_Potencia_Black[250-Wbins:250+Wbins, :]
Potencia_estimada_Black = np.sum(sub_matriz_Black, axis = 0)
Amplitud_Integral_Black = np.sqrt(2*Potencia_estimada_Black)

#FlatTop
Amplitud_Slice_Flat = 2*np.abs(XX_Flat[250 , :])
Densidad_Potencia_Flat = 2*np.abs(XX_Flat)**2
sub_matriz_Flat = Densidad_Potencia_Flat[250-Wbins:250+Wbins, :]
Potencia_estimada_Flat = np.sum(sub_matriz_Flat, axis = 0)
Amplitud_Integral_Flat = np.sqrt(2*Potencia_estimada_Flat)

# vstack concatena verticalmente
Estimadores_rect = np.vstack([Amplitud_Slice_Rect, Amplitud_Integral_Rect]).transpose()
Estimadores_Bart = np.vstack([Amplitud_Slice_Bart, Amplitud_Integral_Bart]).transpose()
Estimadores_Hann = np.vstack([Amplitud_Slice_Hann, Amplitud_Integral_Hann]).transpose()
Estimadores_Black = np.vstack([Amplitud_Slice_Black, Amplitud_Integral_Black]).transpose()
Estimadores_Flat = np.vstack([Amplitud_Slice_Flat, Amplitud_Integral_Flat]).transpose()

# rectangular
Medianas_Rect = np.median(Estimadores_rect, axis = 0)
Sesgo_Rect = np.median(Estimadores_rect, axis = 0) - Amplitud
Varianza_Rect = np.mean((Estimadores_rect - Medianas_Rect)**2, axis = 0)

# Bartlett
Medianas_Bart = np.median(Estimadores_Bart, axis = 0)
Sesgo_Bart = np.median(Estimadores_Bart, axis = 0) - Amplitud
Varianza_Bart = np.mean((Estimadores_Bart - Medianas_Bart)**2, axis = 0)

# Hann
Medianas_Hann = np.median(Estimadores_Hann, axis = 0)
Sesgo_Hann = np.median(Estimadores_Hann, axis = 0) - Amplitud
Varianza_Hann = np.mean((Estimadores_Hann - Medianas_Hann)**2, axis = 0)

# Blackman
Medianas_Black = np.median(Estimadores_Black, axis = 0)
Sesgo_Black = np.median(Estimadores_Black, axis = 0) - Amplitud
Varianza_Black = np.mean((Estimadores_Black - Medianas_Black)**2, axis = 0)

#FlatTop
Medianas_Flat = np.median(Estimadores_Flat, axis = 0)
Sesgo_Flat = np.median(Estimadores_Flat, axis = 0) - Amplitud
Varianza_Flat = np.mean((Estimadores_Flat - Medianas_Flat)**2, axis = 0)

#Agrupo por estimador
Est_Slice = np.vstack([Amplitud_Slice_Rect, Amplitud_Slice_Bart,Amplitud_Slice_Hann,Amplitud_Slice_Black, Amplitud_Slice_Flat]).transpose()
Est_integral = np.vstack([Amplitud_Integral_Rect, Amplitud_Integral_Bart, Amplitud_Integral_Hann, Amplitud_Integral_Black, Amplitud_Integral_Flat]).transpose()

# Histogramas
# plt.clf()
# plt.figure(1)
plt.title("Slice")
kwargs = dict(alpha=0.5,bins = 10, density=False, stacked=True)
kwargs_2 = dict(alpha=0.5, bins = 2,density=False, stacked=True)
plt.hist(Est_Slice[:,0],**kwargs, label = "Rectangular")
plt.hist(Est_Slice[:,1],**kwargs, label = "Bartlett")
plt.hist(Est_Slice[:,2],**kwargs, label = "Hann")
plt.hist(Est_Slice[:,3],**kwargs, label = "Blackman")
plt.hist(Est_Slice[:,4],**kwargs_2, label = "FlatTop")
plt.legend()

plt.figure(2)
plt.title("Integral")
kwargs = dict(alpha=0.5,bins = 10, density=False, stacked=True)
kwargs_2 = dict(alpha=0.5, bins = 2,density=False, stacked=True)
plt.hist(Est_integral[:,0],**kwargs, label = "Rectangular")
plt.hist(Est_integral[:,1],**kwargs, label = "Bartlett")
plt.hist(Est_integral[:,2],**kwargs, label = "Hann")
plt.hist(Est_integral[:,3],**kwargs, label = "Blackman")
plt.hist(Est_integral[:,4],**kwargs_2, label = "FlatTop")
plt.legend()


#                 SESGO, VARIANZA 
tus_resultados = [ 
                   [Sesgo_Rect[0], Varianza_Rect[0]], # <-- Rectangular  Slice 
                   [Sesgo_Rect[1], Varianza_Rect[1]], # <-- Rectangular  Integral
                   [Sesgo_Bart[0], Varianza_Bart[0]], # <-- Bartlett     Slice
                   [Sesgo_Bart[1], Varianza_Bart[1]], # <-- Bartlett     Integral
                   [Sesgo_Hann[0], Varianza_Hann[0]], # <-- Hann         Slice
                   [Sesgo_Hann[1], Varianza_Hann[1]], # <-- Hann         Integral
                   [Sesgo_Black[0], Varianza_Black[0]], # <-- Blackman     Slice
                   [Sesgo_Black[1], Varianza_Black[1]], # <-- Blackman     Integral
                   [Sesgo_Flat[0], Varianza_Flat[0]], # <-- FlatTop      Slice
                   [Sesgo_Flat[1], Varianza_Flat[1]], # <-- FlatTop      Integral
                 ]
df = DataFrame(tus_resultados, columns=['$s_a$', '$v_a$'],
               index=[  
                        'Rectangular Slice',
                        'Rectangular Integral',
                        'Bartlett Slice',
                        'Bartlett Integral',
                        'Hann Slice',
                        'Hann Integral',
                        'Blackman Slice',
                        'Blackman Integral',
                        'Flat-top Slice',
                        'Flat-top Integral'
                     ])
# HTML(df.to_html())


