# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 16:40:32 2022

@author: Nahuel
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

def my_senoidal (N, freq_M, amplitud = 1, valor_medio = 0, freq = 1, fase = 0):
    
    ts = 1/freq_M
    
    tt = np.linspace(0, (N-1)*ts, N)
    
    xx = amplitud * np.sin(2*np.pi*(freq)*tt + fase) + valor_medio
    
    return(tt,xx)

def my_ADC(Sr, B, Vref):
    
    #q = 2*Vref/((2**B)-1) # LSB
    q = Vref/(2**(B-1)) # LSB
    
    Sq = q * np.round(Sr/q)
    
    return Sq
    
fs = 10000 # frecuencia de muestreo
N = 10000   # cantidad de muestras
freq = 1
DC = 2

# ADC
B = 2# Bits
Vref = 1 # Voltaje de referencia
q = Vref/(2**(B-1)) #LSB

[t, Sr] = my_senoidal(N,fs,Vref,DC,freq) # creo mi señal senoidal

sigma = q/2
 
noise = sigma * np.random.randn(len(t))

# noise = 0

Sr = Sr + noise

Sq = my_ADC(Sr,B,Vref)

e = (Sq - Sr)/q # Error normalizado por q

# Ploteo de la señal del ADC
# fig, ax = plt.subplots(nrows=1 , ncols=1)
# # fig.set_size_inches(16,12)
# ax.plot(t,Sr,linewidth = 3)
# ax.plot(t,Sq, color = 'red',linestyle='dashed',linewidth = 3)
# ax.set_ylabel('Amplitud [V]')
# ax.set_xlabel('Tiempo [Seg]')
# ax.grid()

# Histograma
# fig, ax = plt.subplots(nrows=1 , ncols=1)
# fig.set_size_inches(16,12)
# ax.plot(t,e,linewidth = 3)
# ax.hist(e, density=True) # con density = true esta normalizado
# ax.set_title('Histograma normalizado a q ={}'.format(q))
# ax.set_ylabel('Amplitud')
# ax.set_xlabel('q')
# ax.set_xticks([-0.5, -0.4, -0.2, -0.0, 0.2, 0.4, 0.5])
# ax.set_xticklabels(['$\dfrac{-q}{2}$','-0.4','-0.2','0','0.2','0.4','$\dfrac{q}{2}$'])
# ax.grid()
# ax.acorr(e)

# fig, ax = plt.subplots(nrows=1 , ncols=1)
# ax.acorr(e, usevlines=True, normed=True, maxlags = 1000, lw=2)
# ax.grid(True)
print("La media de la señal de error es de {:.2f}".format(np.mean(e)))
print("El desvio estandar de la error es de {:.5f}".format(np.std(Sr)))

print("La varianza de la señal de error es de {:.5f}\n".format(np.var(e)))
Potencia = sum(e**2)/N
print("La Potencia de la señal de error es de {:.5f}".format(Potencia))
print("El valor RMS de mi señal de entrada es de {:.5f}".format(Vref/np.sqrt(2)))