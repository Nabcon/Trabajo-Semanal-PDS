# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 21:31:13 2022

@author: Nahuel
"""
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.signal as sig

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
## generalmente el ancho de banda es de 35 HZ

# del analisis de los latidos
# w1s = 0.25hz
# w1p = 0.75hz
# w2p = 30Hz
# w2s = 50Hz


#%% Analisis en distintas secciones de la ECG

## Ejercicio
ECG_reposo_ejercicio = (ecg_one_lead[450000:550000])
[f, PXX_pot_ejercicio] = sig.welch(ECG_reposo_ejercicio, fs = fs,nperseg = n[4], axis = 0)
Energia_acu = np.cumsum(PXX_pot_ejercicio)

index_Energia_ejercicio = np.where(np.cumsum(PXX_pot_ejercicio)/Energia_acu[-1] > Proporcion)[0]
W_corte_reposo = f[index_Energia_ejercicio[0]]

## pico de esfuerzo
ECG_peak = (ecg_one_lead[750000:850000])
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
#%% Analisis de latidos de ancho fijo
qrs_detections = mat_struct['qrs_detections']

# Viendo el grafico podemos ver que el latido va desde 0 a 600
# qsr me da el pico del latido que se encuentra en la primera muestra en 250
# por eso si me quiero quedar con toda la informacion desde el ecg selecciono desde
# (pico - 250) hasta (pico + 350)
inferior = 250
sup = 350

latido = (ecg_one_lead[int(qrs_detections[0] - inferior):int(qrs_detections[0] + sup)])
muestras = np.arange(len(qrs_detections))

latidos = np.zeros([sup+inferior, qrs_detections.shape[0]])


for nn in muestras:
    latidos[:,nn] = ecg_one_lead[int(qrs_detections[nn] - inferior):int(qrs_detections[nn] + sup)].flatten()
    latidos[:,nn]  -= np.mean(latidos[:,nn]) # le resto su valor medio para centrarlos


# Analizo la muestra de latidos para separar lo normales de los ventriculares
# veo que en la muestra 242 puedo distinguir bien ambos casos
Estimador_amplitud = latidos[242, :]

# Los que estan por debajo de 11500 son latidos normales
# Caso contrario pertenecen a la categoria de ventriculares
filtro_normal = Estimador_amplitud < 11500 #vector booleano

# Index_Ventricular = np.where(Estimador_amplitud > 11500)[0]
# Index_Ventricular = np.where(Estimador_amplitud < 11500)[0]

#Forma rapidisima
# Selecciono del vector de latidos cuales son los que no superan la condicion
Ventricular = latidos[:,np.bitwise_not(filtro_normal)] 
Normal = latidos[:,filtro_normal]

Ventricular_promedio = np.mean(Ventricular, axis = 1)
Normal_promedio = np.mean(Normal, axis = 1)

plt.figure(1)
plt.plot(Ventricular_promedio, 'b', label = 'Ventricular',alpha = 0.5, linewidth=3.0)
plt.plot(Normal_promedio, 'g', label = 'Normal', alpha = 0.5,  linewidth=3.0)
plt.legend()
plt.grid()
# plt.figure(2)
# plt.hist(Estimador_amplitud)

# plt.plot(latidos)