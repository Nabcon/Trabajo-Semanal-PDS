# -*- coding: utf-8 -*-

"""
Created on Wed Aug 31 21:11:21 2022

@author: Nahuel
"""

import numpy as np
import matplotlib.pyplot as plt



def my_senoidal (N, freq_M, amplitud = 1, valor_medio = 0, freq = 1, fase = 0):
    
    ts = 1/freq_M
    
    tt = np.linspace(0, (N-1)*ts, N)
    
    xx = amplitud * np.sin(2*np.pi*(freq)*tt + fase) + valor_medio
    
    return(tt,xx)


def mi_funcion_DFT(xx):

    N = len(xx) #obtengo el largo del vector
    n = np.arange(N) #armo el vector n
    
    k = n.reshape((N, 1)) #armo el vector columna K
    
    #Se realizara todo el recorrido de n por cada fila de K.
    exponent = np.exp(-2j*np.pi*k*(n/N)) #armo la parte exponencial
    
    XX = np.dot(exponent, xx) 
    #realizo la multiplicacion entre mi vector de datos de entrada y mi vector exponencial
    
    return XX

N = 10
freq_M = 10
amplitud = 1
valor_medio = 0
freq = 3
fase = np.pi*0

[t, y] = my_senoidal(N,freq_M, amplitud, valor_medio, freq, fase)

XX = mi_funcion_DFT(y)
XX_py = np.fft.fft(y)
n = np.arange(N)
Res_esp = freq_M/N
f = n*Res_esp

(markers, stemlines, baseline) = plt.stem(f,abs(XX),label='calculado')
#plt.gcf().set_size_inches(16, 12) ASI SE CONFIGURA EL TAMAÑO EN STEM

plt.setp(markers, marker='D', markersize=8, markeredgecolor="orange", markeredgewidth=2)
plt.setp(baseline, color="grey", linestyle="--" )
plt.setp(stemlines, color="purple", linewidth=4)

(markers, stemlines, baseline) = plt.stem(f,abs(XX_py),label='Python FFT',basefmt=" ")
plt.legend()
plt.grid()
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud DFT')

#la verision de jupyter esta mas fachera
