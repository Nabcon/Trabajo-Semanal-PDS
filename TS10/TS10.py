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

[f, PXX_pot_reposo] = sig.welch(ECG_reposo, fs = fs,nperseg = n[5], axis = 0)

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

#%%

def group_delay(ww, phase):
    
    groupDelay = -np.diff(phase)/np.diff(ww)
    
    return(np.append(groupDelay, groupDelay[-1]))

fs = 1000 # Hz
nyq_frec = fs / 2

ripple = 0.05 # dB
alfa_max = 20 # dB
wN=int(5*10e3)

ws1 = 0.25 #Hz
wp1 = 0.75 #Hz
wp2 = 30 #Hz
ws2 = 50 #Hz

len_zeros = int(10e3)

frecs = np.array([0.0,         ws1,         wp1,     wp2,     ws2,         nyq_frec   ]) / nyq_frec
gains = np.array([-alfa_max, -alfa_max, -ripple, -ripple, -alfa_max, -alfa_max])
gains = 10**(gains/20)

bp_sos_iir = sig.iirdesign([wp1, wp2], [ws1, ws2], ripple, alfa_max, analog=False,ftype= 'cheby2', output='sos', fs=fs)

w,h_filter = sig.sosfreqz(bp_sos_iir,worN=wN,whole=False,fs=fs)

# t, y_out = sig.impulse(sig.sos2zpk(bp_sos_iir)) #diverge el impulso

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

# aplico el filtro a mi se;al de ECG

signal_filter = sig.sosfiltfilt(bp_sos_iir, ecg_one_lead, axis = 0)

plt.plot(signal_filter)

#%%
fs = 1000
Proporcion = 0.99

ECG_reposo = (signal_filter[0:100000])

n = [100, 300, 760, 1000, 4000]

[f, PXX_pot_reposo] = sig.welch(ECG_reposo, fs = fs,nperseg = n[4], axis = 0)

Energia_acu = np.cumsum(PXX_pot_reposo)

index_Energia = np.where(np.cumsum(PXX_pot_reposo)/Energia_acu[-1] > Proporcion)[0]

W_corte = f[index_Energia[0]]

ECG_reposo_ejercicio = (signal_filter[450000:550000])
[f, PXX_pot_ejercicio] = sig.welch(ECG_reposo_ejercicio, fs = fs,nperseg = n[4], axis = 0)
Energia_acu = np.cumsum(PXX_pot_ejercicio)

index_Energia_ejercicio = np.where(np.cumsum(PXX_pot_ejercicio)/Energia_acu[-1] > Proporcion)[0]
W_corte_reposo = f[index_Energia_ejercicio[0]]

## pico de esfuerzo
ECG_peak = (signal_filter[750000:850000])
[f, PXX_pot_peak] = sig.welch(ECG_peak, fs = fs,nperseg = n[4], axis = 0)
Energia_acu = np.cumsum(PXX_pot_peak)

index_Energia_peak = np.where(np.cumsum(PXX_pot_peak)/Energia_acu[-1] > Proporcion)[0]
W_corte_peak = f[index_Energia_peak[0]]

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
plt.legend()


#%% graficos
# plt.plot(t,y_out)
plt.subplot(3, 1, 1)
db = 20*np.log10(np.maximum(np.abs(h_filter), 1e-5))
plt.plot(w, db,label='Módulo')
plot_plantilla(filter_type = 'bandpass', fpass = frecs[[2, 3]]* nyq_frec, ripple = ripple , fstop = frecs[ [1, 4] ]* nyq_frec, attenuation = alfa_max, fs = fs)
plt.ylabel('Amplitud [dB]')
plt.ylim(-100, 5)
# plt.xlim(lim1, lim2)
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(w, np.angle(h_filter),label='Fase')
plt.yticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi],
            [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
plt.ylabel('Phase [rad]')
# plt.xlim(lim1, lim2)
plt.show()
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(w,gd,label='Retardo')
plt.ylabel('Tiempo [seg]')
plt.xlabel('Frecuencia [Hz]')
plt.grid(True)
plt.legend()
plt.ylim([0, 1])
