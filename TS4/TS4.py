#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 20:04:17 2022

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

def my_ADC(Sr, B, Vref):
    
    #q = 2*Vref/((2**B)-1) # LSB
    q = Vref/(2**(B-1)) # LSB
    
    Sq = q * np.round(Sr/q)
    
    return Sq

# escalo el numero de muestras y la señal. Por cada 4 muestras de mi señal senoidal, elijo 1
Os = 4
    
fs_Os = Os*1000 # frecuencia de muestreo
N_Os = Os*1000   # cantidad de muestras
freq = 1
DC = 0
Amplitud = 1

# ADC
B = 4 # Bits
Vref = 2 # Voltaje de entrada del ADC
q = Vref/(2**(B-1)) #LSB

# Especificaciones
SNR = 40

# Potencia de ruido
kn = 1
pot_ruido = q**2/12 * kn

[t, Sr] = my_senoidal(N_Os,fs_Os,Amplitud, DC,freq) # creo mi señal senoidal

# Para tener un piso de ruido analogico de 60db, tengo que encontrar que sigma
# me cumple con eso

Sigma_2 = Amplitud**2/(2*10**(SNR/10))

noise_a = np.sqrt(Sigma_2) * np.random.randn(len(Sr))

Sr_ruido = Sr + noise_a # le sumo el ruido

# Realizo el oversampling para poder simular el comportamiento de no tener infinitas muestras
N = N_Os/Os
fs = fs_Os/Os

S_D = np.arange(N)
S_D = Sr_ruido[::Os] #Inicio : Final : Cada cuantas muestras

Sq = my_ADC(S_D,B,Vref) #cuantizo mi señal sin el oversampling con ruido


noise_d = (Sq - S_D)/q  # Ruido digital


# Creo los vectores para graficar
ff = np.arange(0, fs, fs/N)
ff_os = np.arange(0, fs_Os, fs_Os/N_Os)

# SNR_f = 10*np.log10(Amplitud**2/(2*Sigma_2)) ## verifico que obtuve mi SNR

f_noise_a = np.fft.fft(noise_a)

f_noise_d = np.fft.fft(noise_d)

# ax.plot(k_D/2, abs(f_noise_a)**2)

## armo el vector para las transformada de fouriere


# ax.plot(t,Sr_ruido,linewidth = 3)

fig, ax = plt.subplots(nrows=1 , ncols=1)

bfrec = ff <= fs/2


SNR_verif = 10*np.log10(np.var(Sr)/np.var(noise_a))



plt.plot( ff_os[ff_os <= fs/2], 10* np.log10(2*np.abs(f_noise_a[ff_os <= fs/2])**2), ':r')

# plt.figure(2)
 
# Nnq_mean = np.mean(np.abs(ft_Nq)**2)

# plt.plot( ff[bfrec], 10* np.log10(2*np.abs(f_noise_d[bfrec])**2), ':c')
 
# plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Srq[bfrec])**2), lw=2, label='$ s_Q = Q_{B,V_F}\{s_R\} $ (ADC out)' )
# plt.plot( ff_os[ff_os <= fs/2], 10* np.log10(2*np.abs(ft_As[ff_os <= fs/2])**2), color='orange', ls='dotted', label='$ s $ (analog)' )
# plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_SR[bfrec])**2), ':g', label='$ s_R = s + n $  (ADC in)' )
# plt.plot( ff_os[ff_os <= fs/2], 10* np.log10(2*np.abs(ft_Nn[ff_os <= fs/2])**2), ':r')
# plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nq[bfrec])**2), ':c')
# plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ), '--r', label= '$ \overline{n} = $' + '{:3.1f} dB (piso analog.)'.format(10* np.log10(2* nNn_mean)) )
# plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--c', label='$ \overline{n_Q} = $' + '{:3.1f} dB (piso digital)'.format(10* np.log10(2* Nnq_mean)) )
# plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q) )
# plt.ylabel('Densidad de Potencia [dB]')
# plt.xlabel('Frecuencia [Hz]')
# axes_hdl = plt.gca()
# axes_hdl.legend()
# # suponiendo valores negativos de potencia ruido en dB
# plt.ylim((1.5*np.min(10* np.log10(2* np.array([Nnq_mean, nNn_mean]))),10))
 
 
# plt.figure(3)
# bins = 10
# plt.hist(nq, bins=bins)
# plt.plot( np.array([-q/2, -q/2, q/2, q/2]), np.array([0, N/bins, N/bins, 0]), '--r' )
# plt.title( 'Ruido de cuantización para {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Vf, q))
 
 

