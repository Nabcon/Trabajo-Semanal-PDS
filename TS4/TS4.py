#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 20:04:17 2022

@author: nabcon
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.fft import fft, fftshift

def my_senoidal (N, freq_M, amplitud = 1, valor_medio = 0, freq = 1, fase = 0):
    
    ts = 1/freq_M
    
    tt = np.linspace(0, (N-1)*ts, N)
    
    xx = amplitud * np.sin(2*np.pi*(freq)*tt + fase) + valor_medio
    
    return(tt,xx)

def my_ADC(Sr, B, Vref):
    
    q = Vref/(2**(B-1)) # LSB
    Sq = q * np.round(Sr/q)
    error = Sq - Sr
    return Sq, error

def ADC_analisis(kn,fs,N,B,Vref,Amplitud,ff = 1):

    ts = 1/fs
    
    q = Vref/(2**(B-1))
    
    over_sampling = 1
    N_os = N*over_sampling
    fs_os = fs*over_sampling
    

    [tt_os, analog_sig] = my_senoidal(N_os,fs_os,amplitud = Amplitud,freq = ff) # creo mi señal senoidal con OS
    tt = np.linspace(0, (N-1)*ts, N) #vector temporal sin OS
    
    q = Vref/(2**(B-1)) # LSB
    pot_ruido = ((q**2)/12 )* kn # Watts (potencia de la señal 1 W)
    desvio = np.sqrt(pot_ruido)
    
    #Metodo de definir manualmente un piso de ruido analogico
    #SNR = 40
    #Sigma_2 = Amplitud**2/(2*10**(SNR/20))
    #noise_a = np.sqrt(Sigma_2) * np.random.randn(len(analog_sig))
    # SNR_f = 10*np.log10(Amplitud**2/(2*Sigma_2)) ## verifico que obtuve mi SNR
    
    # Ruido incorrelado y gaussiano
    n = np.random.normal(0, desvio, size=N_os)
    #signal_ruido = analog_sig + n
    signal_ruido = analog_sig + n
    
    sr = signal_ruido[::over_sampling] #le saco el OS a mi señal ruidosa
    
    #ADC
    srq, nq= my_ADC(sr,B,Vref)
    
    # Densidad espectral de potencia
    ff = np.arange(0, fs, fs/N)
    ff_os = np.arange(0, fs_os, fs_os/N_os)
    ft_Srq = fft(srq,N)/srq.shape[0]
    ft_As = fft(analog_sig,N_os)/analog_sig.shape[0]
    ft_SR = fft(sr,N)/sr.shape[0]
    ft_Nn = fft(n,N_os)/n.shape[0]  #piso de ruido analogico -> ruido gausseano
    ft_Nq = fft(nq,N)/nq.shape[0]   #piso de ruido digital -> error cuantizacion
    
    bfrec = ff <= fs/2
    nNn_mean = np.mean(np.abs(ft_Nn)**2)
    Nnq_mean = np.mean(np.abs(ft_Nq)**2)
    
    plt.close('all')
     
    plt.figure(1)
    plt.plot(tt, srq, lw=2, label='$ s_Q = Q_{B,V_F}\{s_R\} $ (ADC out)')
    plt.plot(tt, sr, linestyle=':', color='green',marker='o', markersize=3, markerfacecolor='none', markeredgecolor='green', fillstyle='none', label='$ s_R = s + n $  (ADC in)')
    plt.plot(tt_os, analog_sig, color='orange', ls='dotted', label='$ s $ (analog)')
    plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Amplitud, q) )
    plt.xlabel('tiempo [segundos]')
    plt.ylabel('Amplitud [V]')
    axes_hdl = plt.gca()
    axes_hdl.legend()
    plt.show()
     
    plt.figure(2) 
    plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Srq[bfrec])**2), lw=2, label='$ s_Q = Q_{B,V_F}\{s_R\} $ (ADC out)' )
    plt.plot( ff_os[ff_os <= fs/2], 10* np.log10(2*np.abs(ft_As[ff_os <= fs/2])**2), color='orange', ls='dotted', label='$ s $ (analog)' )
    plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_SR[bfrec])**2), ':g', label='$ s_R = s + n $  (ADC in)' )
    plt.plot( ff_os[ff_os <= fs/2], 10* np.log10(2*np.abs(ft_Nn[ff_os <= fs/2])**2), ':r')
    plt.plot( ff[bfrec], 10* np.log10(2*np.abs(ft_Nq[bfrec])**2), ':c')
    plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ), '--r', label= '$ \overline{n} = $' + '{:3.1f} dB (piso analog.)'.format(10* np.log10(2* nNn_mean)) )
    plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--c', label='$ \overline{n_Q} = $' + '{:3.1f} dB (piso digital)'.format(10* np.log10(2* Nnq_mean)) )
    plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Amplitud, q) )
    plt.ylabel('Densidad de Potencia [dB]')
    plt.xlabel('Frecuencia [Hz]')
    axes_hdl = plt.gca()
    axes_hdl.legend()
    # suponiendo valores negativos de potencia ruido en dB
    plt.ylim((1.5*np.min(10* np.log10(2* np.array([Nnq_mean, nNn_mean]))),10))
     
     
    plt.figure(3)
    bins = 10
    plt.hist(nq, bins=bins)
    plt.plot( np.array([-q/2, -q/2, q/2, q/2]), np.array([0, N/bins, N/bins, 0]), '--r' )
    plt.title( 'Ruido de cuantización para {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B, Amplitud, q))

    return

#%%
kn = 1
fs = 1000
N = 1000
B = 4
Vref = 1
A = 2

ADC_analisis(kn,fs,N,B,Vref,A)

#%%
#Analizar para una de las siguientes configuraciones B = ̣{4, 8 y 16} bits, kn={1/10,1,10}
#Discutir los resultados respecto a lo obtenido en a).
B = 4
kn = 1/10
ADC_analisis(kn,fs,N,B,Vref,A)

#%%
B = 4
kn = 10
ADC_analisis(kn,fs,N,B,Vref,A)

#%%
B = 8
kn = 1/10
ADC_analisis(kn,fs,N,B,Vref,A)

#%%
B = 8
kn = 1
ADC_analisis(kn,fs,N,B,Vref,A)

#%%
B = 8
kn = 10
ADC_analisis(kn,fs,N,B,Vref,A)
#%%
B = 16
kn = 1/10
ADC_analisis(kn,fs,N,B,Vref,A)

#%%
B = 16
kn = 1
ADC_analisis(kn,fs,N,B,Vref,A)

#%%
B = 16
kn = 10
ADC_analisis(kn,fs,N,B,Vref,A)

#%% ALIAS

#Nyquist dice que para evitar el efecto de Alias, se tiene que cumplir que
#fs >= 2.ff

# datos de la senoidal
N = 1000  # cantidad de muestras

factor = 0.9 # factor para hacer que fs < 2.ff
frec = 1
fs = 2*frec*factor

B = 4
kn = 1

ADC_analisis(kn,fs,N,B,Vref,A,ff = frec)
