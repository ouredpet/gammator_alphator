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

def calc_RTD(RTD):
    path_load = os.path.join(str(RTD), "j.vs.V.dat")

    data = np.genfromtxt(path_load, delimiter='\t')

    i_interp_func = interpolate.interp1d(data[:,0],data[:,1],'cubic',fill_value="extrapolate")

    V = np.linspace(data[0,0],data[-1,0],10001)
    i = i_interp_func(V)

    dV = find_delta_V(V, i)

    # Calculation of the large signal gain g^(1) first harmonic
    Vac = np.linspace(1e-4,1,201)
    Vb = np.linspace(0.5,1.2,201)
    g1_max = calculate_g1_max(Vac, Vb, i_interp_func)

    # # cut only for negative gains
    # Vac_2 = Vac[g1_max<=0]
    # g1_max2 = g1_max[g1_max<=0]

    # gamma
    gamma = g1_max/g1_max[0]

    # alpha calculation from the definition
    alpha = 1/((1-gamma))*(Vac / dV)**2
    # in case gamma = 1, alpha is irelevant
    alpha[gamma == 1] = 0

    path_save = os.path.join("data_out", "alpha" + str(RTD) + ".npy")
    np.save(path_save, np.array((gamma, alpha)))

    path_save = os.path.join("data_out", "gamma" + str(RTD) + ".npy")
    np.save(path_save, np.array((Vac, gamma)))

    path_save = os.path.join("data_out", "g1_max" + str(RTD) + ".npy")
    np.save(path_save, np.array((Vac, g1_max)))

    return dV, V, i, Vac, g1_max, gamma, alpha


# for spyder
plt.close('all')

# choose data
RTD_16 = 16
RTD_10 = 10


dV_10, V_10, i_10, Vac_10, g1_max_10, gamma_10, alpha_10 = calc_RTD(RTD_10)
dV_16, V_16, i_16, Vac_16, g1_max_16, gamma_16, alpha_16 = calc_RTD(RTD_16)




####################

fig1, ax1 = plt.subplots()
ax1.plot(V_10,i_10, label="10")
ax1.plot(V_16,i_16, label="16")
ax1.grid()
ax1.set_xlabel("Voltage (V)")
ax1.set_ylabel("Current density (mA/um2)")
fig1.tight_layout()



####################

fig1, ax1 = plt.subplots()
ax1.plot(Vac_10,g1_max_10)
ax1.plot(Vac_16,g1_max_16)
ax1.grid()
ax1.set_xlabel("Voltage Amplitude Vac (V)")
ax1.set_ylabel("LS Conductance envelope (mS/um2)")
fig1.tight_layout()

fig1, ax1 = plt.subplots()
ax1.plot(Vac_10**2,g1_max_10)
ax1.plot(Vac_16**2,g1_max_16)
ax1.grid()
ax1.set_xlabel("Voltage Amplitude Vac squared (V^2)")
ax1.set_ylabel("LS Conductance envelope (mS/um2)")
ax1.set_xscale("log")
# ax1.set_yscale("log")
fig1.tight_layout()


fig1, ax1 = plt.subplots()
ax1.plot(g1_max_10, Vac_10**2)
ax1.plot(g1_max_16, Vac_16**2)
ax1.grid()
ax1.set_xlabel("LS Conductance envelope (mS/um2)")
ax1.set_ylabel("Voltage Amplitude Vac squared (V^2)")
ax1.set_yscale("log")
# ax1.set_xscale("log")
fig1.tight_layout()


# fig1, ax1 = plt.subplots()
# ax1.plot(1/2*np.convolve(Vac_10**2,(1,1),"valid"),np.diff(g1_max_10)/np.diff(Vac_10**2))
# ax1.plot(1/2*np.convolve(Vac_16**2,(1,1),"valid"),np.diff(g1_max_16)/np.diff(Vac_16**2))
# ax1.grid()
# ax1.set_xlabel("Voltage Amplitude Vac squared (V^2)")
# ax1.set_ylabel("LS Conductance envelope (mS/um2)")
# ax1.set_xscale("log")
# ax1.set_yscale("log")
# fig1.tight_layout()


#############
fig1, ax1 = plt.subplots()
ax1.plot(gamma_10,alpha_10)
ax1.plot(gamma_16,alpha_16)
ax1.grid()
ax1.set_xlabel("gama (-)")
ax1.set_ylabel("alpha (-)")
fig1.tight_layout()




plt.show()



