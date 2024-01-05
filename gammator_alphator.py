# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 15:11:08 2023

@author: pourednik
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import os

def find_delta_V(V, i):
    # Here I seachr for delta_V of the iv curve
    didV = np.diff(i)/np.diff(V)
    Vd = 1/2*np.convolve(V, (1,1), 'valid')

    # search for indexes of of the peak and valley voltage
    # where the first derivate is zero
    dV0i = np.where(np.diff(np.sign(didV)))[0]
    # find delta_V
    delta_V = np.diff(Vd[dV0i])
    return delta_V

def calculate_g1_max(Vac, Vb, i_interp_func):
    phase = np.linspace(0,2*np.pi-1/1001,1001) # needs to be exactly 2*pi-1/samplenumber
    v_n = np.cos(phase)
    g1_max = 100*np.ones((Vac.size)) # very large value over which i compare

    for i_Vac in range(Vac.size):
        for i_Vb in range(Vb.size):
            # current in the time domain
            i = i_interp_func(Vac[i_Vac]*v_n + Vb[i_Vb])

            # first harmonic of the real value of the current
            # two because of the negative frequency line
            I1 = 2 * (np.real(np.fft.fft(i))/i.size)[1]
            g1 = I1 / Vac[i_Vac]

            # envelop search
            if g1_max[i_Vac]>g1:
                g1_max[i_Vac] = g1

    return g1_max



# for spyder
plt.close('all')

# choose data
RTD = 16
# RTD = 10

path_load = os.path.join(str(RTD), "j.vs.V.dat")



data = np.genfromtxt(path_load, delimiter='\t')

i_interp_func = interpolate.interp1d(data[:,0],data[:,1],'cubic',fill_value="extrapolate")

V = np.linspace(data[0,0],data[-1,0],10001)
i = i_interp_func(V)

# show iv and its fit
fig1, ax1 = plt.subplots()
ax1.plot(data[:,0],data[:,1], label="Raw")
ax1.plot(V,i, label="Fit")
ax1.grid()
ax1.set_xlabel("Voltage (V)")
ax1.set_ylabel("Current density (mA/um2)")
fig1.tight_layout()


################

dV = find_delta_V(V, i)


#########
# Calculation of the large signal gain g^(1) first harmonic

Vac = np.linspace(1e-4,1,201)
Vb = np.linspace(0.5,1.2,201)

g1_max = calculate_g1_max(Vac, Vb, i_interp_func)
####################

fig1, ax1 = plt.subplots()
ax1.plot(Vac,g1_max)
ax1.grid()
ax1.set_xlabel("Voltage Amplitude Vac (V)")
ax1.set_ylabel("LS Conductance envelope (mS/um2)")
fig1.tight_layout()

# cut only for negative gains
Vac_2 = Vac[g1_max<=0]
g1_max2 = g1_max[g1_max<=0]

# gamma
gamma = g1_max2/g1_max2[0]

# alpha calculation from the definition
alpha = 1/((1-gamma))*Vac_2**2/dV**2
# in case gamma = 1, alpha is irelevant
alpha[gamma == 1] = 0

fig1, ax1 = plt.subplots()
ax1.plot(gamma,alpha)
ax1.grid()
ax1.set_xlabel("gama (-)")
ax1.set_ylabel("alpha (-)")
fig1.tight_layout()


path_save = os.path.join("data_out", "alpha" + str(RTD) + ".npy")
np.save(path_save, np.array((gamma, alpha)))

path_save = os.path.join("data_out", "gamma" + str(RTD) + ".npy")
np.save(path_save, np.array((Vac_2, gamma)))

path_save = os.path.join("data_out", "g1_max" + str(RTD) + ".npy")
np.save(path_save, np.array((Vac_2, g1_max2)))

plt.show()



