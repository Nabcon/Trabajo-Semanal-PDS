# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 21:31:49 2023

@author: Nahuel
"""
import scipy.signal as sig
import numpy as np
import matplotlib as mpl
#import matplotlib.pyplot as plt
import scipy.io as sio
from splane import plot_plantilla

def group_delay(ww, phase):
    
    groupDelay = -np.diff(phase)/np.diff(ww)
    
    return(np.append(groupDelay, groupDelay[-1]))

fs = 1000 # Hz
nyq_frec = fs / 2

# filter design
ripple = 0 # dB
atenuacion = 40 # dB

# ws1 = 0.1 #Hz
# wp1 = 1.0 #Hz
# wp2 = 40.0 #Hz
# ws2 = 50.0 #Hz

ws1 = 0.25 #Hz
wp1 = 0.75 #Hz
wp2 = 30 #Hz
ws2 = 50 #Hz

# vector de frecuencias de interes normalizadas por nyq
# es la estructura para un pasabanda
frecs = np.array([0.0,         ws1,         wp1,     wp2,     ws2,         nyq_frec   ]) / nyq_frec

# me armo un vector de ganancias asociado a cada banda de frecuencias
gains = np.array([-atenuacion, -atenuacion, -ripple, -ripple, -atenuacion, -atenuacion])

#lo paso a veces
gains = 10**(gains/20)


#% CASO PASA ALTO

# sacamos del vector de frecuencias lo necesario para el pasa alto
# recordad que el vector frecs se encuentra normalizado por nyq
# como la estructurna necesaria es: [Inicio = 0, ws, wp, nyq = 1]
# agrego con la funcion append el 1.0 que representa el final:nyq
ls_bands_hp = np.append(frecs[:3],[1.0])

# Necesitamos la ganancia deseada en cada una de las bandas seleccionadas anterioremente
# Armo la el vector de acuerdo a lo necesitado (NORMALIZADO)
# [inicio -> -40db  , ws -> 40db, wp, -> 0db, nyq -> 0db]
ls_desired_hp = np.append(gains[:3],[1.0])

#tiene que tener la mitad del largo del vector de bandas (PREGUNTAR)
ls_weight_hp = np.array([20, 1])

cant_coef = 8000
# firls pide que la cantidad de coeficientes sean impares por lo que
# si la cantidad de coeficientes es par,  le sumo 1
if cant_coef % 2 == 0:
    cant_coef += 1

fs_n = fs/nyq_frec

num_firls_hp = sig.firls(cant_coef, ls_bands_hp, ls_desired_hp, weight = ls_weight_hp, fs=fs_n)

#% CASO PASA BAJO 
#            0    1    2    3    4     5
# frecs = ([0.0, ws1, wp1, wp2, ws2,nyq_frec])

# estructura pasa bajo de frecuencias de interes  
# [Inicio = 0, wp2, ws2, nyq]
ls_band_lp = np.append([0.0],frecs[3:])

# [inicio -> 0db  , wp -> 0db, ws, -> 40db, nyq -> 40db]
ls_desired_lp = np.append([1.0],gains[3:])

#tiene que tener la mitad del largo del vector de bandas (PREGUNTAR)
ls_weight_lp = np.array([5, 10])

cant_coef = 500
# firls pide que la cantidad de coeficientes sean impares por lo que
# si la cantidad de coeficientes es par,  le sumo 1
if cant_coef % 2 == 0:
    cant_coef += 1

num_firls_lp = sig.firls(cant_coef, ls_band_lp, ls_desired_lp, weight = ls_weight_lp, fs=fs_n)

#% UNION PASABANDA
import matplotlib.pyplot as plt

# Para poder apreciar bien la señal, necesito tener muchos puntos en la zona de interes
# Por divido a mi vector de w en dos partes. La parte donde quiero varios puntos, de 0  a 40 
# lo creo logaritmado. La siguiente parte dede 40 a 500 lo hago con espaciamiento lineal
w_rad  = np.append(np.logspace(-2, 0.8, 250), np.logspace(0.9, 1.6, 250) )
w_rad  = np.append(w_rad, np.linspace(40, nyq_frec, 500, endpoint=True) ) / nyq_frec * np.pi
w = w_rad / np.pi * nyq_frec

num_firls_bp = np.convolve(num_firls_lp, num_firls_hp)

# en un filtro del tipo FIR el denominador es siempre unitario.
den_fir = 1.0


_, h_firls_bp = sig.freqz(num_firls_bp, den_fir,w_rad)

plt.plot(w, 20 * np.log10(abs(h_firls_bp)), label='FIR-ls')
plot_plantilla(filter_type = 'bandpass', fpass = frecs[[2, 3]]* nyq_frec, ripple = ripple , fstop = frecs[ [1, 4] ]* nyq_frec, attenuation = atenuacion, fs = fs)

# plt.figure(2)

# phase_firls = np.angle(h_firls_bp)
# plt.plot(w, phase_firls, label='FIR-ls')    # Bode phase plot

# plt.figure(3)
# gd_firls = group_delay(w_rad, phase_firls)
# plt.plot(w[gd_firls>0], gd_firls[gd_firls>0], label='FIR-ls')    # Bode phase plot
# plt.ylim([-1,1])

#%% Arranco con la parte del ecg

# Setup inline graphics
mpl.rcParams['figure.figsize'] = (10,10)

# para listar las variables que hay en el archivo
#io.whosmat('ecg.mat')
mat_struct = sio.loadmat('ecg.mat')

ecg_one_lead = mat_struct['ecg_lead']
ecg_one_lead = ecg_one_lead.flatten()
cant_muestras = len(ecg_one_lead)
