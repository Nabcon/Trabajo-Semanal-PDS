# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 23:05:48 2022

@author: Nahuel
"""

import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
from splane import plot_plantilla
import scipy.io as sio

mat_struct = sio.loadmat('ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead']
plt.figure(1)
#ploteo una la señal
plt.plot(ecg_one_lead)
plt.grid()
# plt.xlim([0,100000])

fs = 1000
Proporcion = 0.99

ECG_reposo = (ecg_one_lead[0:100000])

# ECG_reposo = ECG_reposo.astype(np.float64)

# ECG_reposo -= np.mean(ECG_reposo) # le saco el valor medio


## la señal se repite desde 0 a 765
n = [100, 300, 760, 1000, 4000]
i=0

# evaluo con que nperseg me quedo
# for nn in n:
#     [f, PXX_pot] = sig.welch(ECG_reposo, fs = fs,nperseg = nn, axis = 0)
#     plt.plot(f,10*np.log10(2*(PXX_pot)), label='%i' % n[i])
#     i=i+1
    
## elijo nperseg = 760

[f, PXX_pot_reposo] = sig.welch(ECG_reposo, fs = fs,nperseg = n[4], axis = 0)

# plt.legend()
# plt.grid()

Energia_acu = np.cumsum(PXX_pot_reposo)

index_Energia = np.where(np.cumsum(PXX_pot_reposo)/Energia_acu[-1] > Proporcion)[0]

W_corte = f[index_Energia[0]]

# del analisis de los latidos
# w1s = 0.25hz
# w1p = 0.75hz
# w2p = 30Hz
# w2s = 50Hz

#%% DISEÑO DE FILTRO IIR Y APLICACION
def group_delay(ww, phase):
    
    groupDelay = -np.diff(phase)/np.diff(ww)
    
    return(np.append(groupDelay, groupDelay[-1]))

fs = 1000 # Hz
nyq_frec = fs / 2

# Como utilizaremos FiltFilt,el ripple y la atenuacion final seran el doble de la especificada
ripple = 0.05 # dB
alfa_min = 20 # dB

ws1 = 0.25 #Hz
wp1 = 0.75 #Hz
wp2 = 30 #Hz
ws2 = 50 #Hz

len_zeros = int(10e3)

frecs = np.array([0.0,         ws1,         wp1,     wp2,     ws2,         nyq_frec   ]) / nyq_frec

bp_sos_iir = sig.iirdesign([wp1, wp2], [ws1, ws2], ripple, alfa_min, analog=False,ftype= 'cheby2', output='sos', fs=fs)

w_rad  = np.append(np.logspace(-2, 0.8, 250), np.logspace(0.9, 1.6, 250) )
w_rad  = np.append(w_rad, np.linspace(40, nyq_frec, 500, endpoint=True) ) / nyq_frec * np.pi
w = w_rad / np.pi * nyq_frec

_,h_filter = sig.sosfreqz(bp_sos_iir,worN=w_rad)


# cumplimos con la plantilla
# plt.plot(w, 20 * np.log10(abs(h_filter)), label='FIR-ls')
# plot_plantilla(filter_type = 'bandpass', fpass = frecs[[2, 3]]* nyq_frec, ripple = ripple , fstop = frecs[ [1, 4] ]* nyq_frec, attenuation = alfa_min, fs = fs)

gd = group_delay(w,np.angle(h_filter))

# hago este for para poder borrar los sobrepicos erroneos del retardo de grupo
for i in range(len(gd)-1):
    if (gd[i] > 10) or ( gd[i] < 0): # si detecto esos sobrepicos ficticios
        # realizo un promedio de la muestra anterior y la futura
        gd[i] = (gd[i-1] + gd[i+1])/2  
        
impulso = np.zeros(len_zeros)
impulso[0] = 1

respuesta_impulso = sig.sosfiltfilt(bp_sos_iir, impulso)
tt = np.arange(len_zeros)
# plt.plot(tt, respuesta_impulso)

# aplico el filtro a mi señal de ECG

ECG_f_IIR = sig.sosfiltfilt(bp_sos_iir, ecg_one_lead, axis = 0)

plt.plot(ECG_f_IIR)
plt.plot(ecg_one_lead)
#%% DISEÑO DE FILTRO FIR Y APLICACION
# Utilizo el filtro FIR del tipo cuadrados minimos

ripple = 0 # dB
atenuacion = 20 # dB

fs_n = fs/nyq_frec

# vector de frecuencias necesarias para el diseño
frecs = np.array([0.0,         ws1,         wp1,     wp2,     ws2,         nyq_frec   ]) / nyq_frec

# ganancias deseadas
gains = np.array([-atenuacion, -atenuacion, -ripple, -ripple, -atenuacion, -atenuacion]) # dB
gains = 10**(gains/20) # Veces

# diseño pasa alto
ls_bands_hp = np.append(frecs[:3],[1.0])
ls_desired_hp = np.append(gains[:3],[1.0])
ls_weight_hp = np.array([20, 1])

cant_coef = 3000    # Esta parte es mas exigente, por eso requiero mas coeficientes
if cant_coef % 2 == 0:
    cant_coef += 1
    

num_firls_hp = sig.firls(cant_coef, ls_bands_hp, ls_desired_hp, weight = ls_weight_hp, fs=fs_n)

# diseño pasa bajo
ls_band_lp = np.append([0.0],frecs[3:])
ls_desired_lp = np.append([1.0],gains[3:])
ls_weight_lp = np.array([5, 10])
cant_coef = 500 # Menor exigencia, menor cantidad de coeficientes

if cant_coef % 2 == 0:
    cant_coef += 1

num_firls_lp = sig.firls(cant_coef, ls_band_lp, ls_desired_lp, weight = ls_weight_lp, fs=fs_n)

# diseño pasa banda : Union de pasa alto y pasa bajo
w_rad  = np.append(np.logspace(-2, 0.8, 250), np.logspace(0.9, 1.6, 250) )
w_rad  = np.append(w_rad, np.linspace(40, nyq_frec, 500, endpoint=True) ) / nyq_frec * np.pi
w = w_rad / np.pi * nyq_frec

num_firls_bp = np.convolve(num_firls_lp, num_firls_hp)
den_fir = 1.0

_, h_firls_bp = sig.freqz(num_firls_bp, den_fir,w_rad)

# cumplimos con la plantilla
#plt.plot(w, 20 * np.log10(abs(h_firls_bp)), label='FIR-ls')
#plot_plantilla(filter_type = 'bandpass', fpass = frecs[[2, 3]]* nyq_frec, ripple = ripple , fstop = frecs[ [1, 4] ]* nyq_frec, attenuation = atenuacion, fs = fs)

# aplico el filtro a mi señal de ECG
ECG_f_FIR = sig.filtfilt(num_firls_bp, den_fir, ecg_one_lead,axis = 0)
plt.plot(ECG_f_FIR)
plt.plot(ecg_one_lead)


#%%Comparo ambos filtros
plt.plot(ecg_one_lead, label = "Original")
plt.plot(ECG_f_IIR, label = "IIR")
plt.plot(ECG_f_FIR, label = "FIR")
plt.legend()

#%% Analisis de ECG_IIR en distintas secciones
fs = 1000
n = [100, 300, 760, 1000, 4000]

ECG_reposo = (ECG_f_IIR[0:100000])
[f, PXX_pot_reposo] = sig.welch(ECG_reposo, fs = fs,nperseg = n[4], axis = 0)

ECG_ejercicio = (ECG_f_IIR[450000:550000])
[f, PXX_pot_ejercicio] = sig.welch(ECG_ejercicio, fs = fs,nperseg = n[4], axis = 0)

## pico de esfuerzo
ECG_peak = (ECG_f_IIR[750000:850000])
[f, PXX_pot_peak] = sig.welch(ECG_peak, fs = fs,nperseg = n[4], axis = 0)

plt.figure(2)
plt.plot(f, 10*np.log10(PXX_pot_reposo), label = 'reposo')
plt.plot(f, 10*np.log10(PXX_pot_ejercicio), label = 'ejercicio')
plt.plot(f, 10*np.log10(PXX_pot_peak), label = 'pico')
plt.grid()

plt.legend()
plt.figure(3)
plt.plot(f, (PXX_pot_reposo), label = 'reposo')
plt.plot(f, (PXX_pot_ejercicio), label = 'ejercicio')
plt.plot(f, (PXX_pot_peak), label = 'pico')
plt.grid()
plt.xlim([0,30])
plt.legend()

#%% Analisis de ECG_FIR en distintas secciones
fs = 1000
n = [100, 300, 760, 1000, 4000]

ECG_reposo = (ECG_f_FIR[0:100000])
[f, PXX_pot_reposo] = sig.welch(ECG_reposo, fs = fs,nperseg = n[4], axis = 0)

ECG_ejercicio = (ECG_f_FIR[450000:550000])
[f, PXX_pot_ejercicio] = sig.welch(ECG_ejercicio, fs = fs,nperseg = n[4], axis = 0)

## pico de esfuerzo
ECG_peak = (ECG_f_FIR[750000:850000])
[f, PXX_pot_peak] = sig.welch(ECG_peak, fs = fs,nperseg = n[4], axis = 0)

plt.figure(2)
plt.plot(f, 10*np.log10(PXX_pot_reposo), label = 'reposo')
plt.plot(f, 10*np.log10(PXX_pot_ejercicio), label = 'ejercicio')
plt.plot(f, 10*np.log10(PXX_pot_peak), label = 'pico')
plt.grid()

plt.legend()
plt.figure(3)
plt.plot(f, (PXX_pot_reposo), label = 'reposo')
plt.plot(f, (PXX_pot_ejercicio), label = 'ejercicio')
plt.plot(f, (PXX_pot_peak), label = 'pico')
plt.grid()
plt.xlim([0,30])
plt.legend()