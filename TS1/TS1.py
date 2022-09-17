# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 18:45:09 2022

GENERADOR DE FUNCIONES SENOIDALES

@author: Nahuel
"""

import numpy as np
import matplotlib.pyplot as plt


#N = 1000
#fs = 1000

def my_senoidal (N, freq_M, amplitud = 1, valor_medio = 0, freq = 1, fase = 0):
    
    ts = 1/freq_M
    
    tt = np.linspace(0, (N-1)*ts, N)
    
    xx = amplitud * np.sin(2*np.pi*(freq)*tt + fase)
    
    return(tt,xx)

[t, y] = my_senoidal(N=100,freq_M = 100)

fig, ax = plt.subplots(nrows=1 , ncols=1)

ax.plot(t,y)