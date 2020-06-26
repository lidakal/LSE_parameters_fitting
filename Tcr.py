import numpy as np
import matplotlib.pyplot as plt
import LidaLib as lil
import pandas as pd
from uncertainties import ufloat

def Evarshni(T, a, b, p):
    Eo, sigma, tau, DE = p

    k = 8.617333262145e-5 # in eV*K^-1
    kT = k * T
    kTheta = k * b
    X = lil.XofT(kT, sigma, tau, DE)
    return Eo - (a * np.power(kT,2) / (k * (kTheta + kT))) - (X * kT)

def find_Tcr(T_arr, E_arr):
    Tcr = 15
    Emin = 100.
    for (T, E) in zip(T_arr, E_arr):
        if E < Emin:
            Emin = E
            Tcr = T
    return Tcr

T_arr = np.linspace(15, 300, 100)

Series = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12'] # Sample identifiers

popt_all = pd.read_csv('C:/Users/letha/Dropbox/pl/EvsT/4varData/results_with_errors.csv')
popt_all = popt_all.set_index('Series')

#sigma_arr = np.array(popt_all.loc['S1':'S12', 'sigma (meV)']) # in meV
#dsigma_arr = np.array(popt_all.loc['S1':'S12', 'dsigma (meV)']) # in meV

Tcr_arr = np.array([])
depth_arr = np.array([])
DE_arr = -np.array(popt_all.loc['S1':'S9', 'Ea - Eo (meV)']) # in meV, to fit without outliers

for ser in Series:
    popt = np.array(popt_all.loc[ser])
    a, b, Eo, dEo, sigma, dsigma, tau, dtau, DE, dDE = popt
    
    # back to correct units
    a = a / 1000.
    sigma = sigma / 1000.
    dsigma = dsigma / 1000.
    DE = - DE / 1000.
    dDE = dDE / 1000.

    p0 = [Eo, sigma, tau, DE]
    E = Evarshni(T_arr, a, b, p0)

    Tcr = find_Tcr(T_arr, E)
    Tcr_arr = np.append(Tcr_arr, Tcr)

    if ser in ['S10', 'S11', 'S12']:
        color = 'tab:red'
    else:
        color = 'tab:blue'
    '''
    plt.plot(Tcr, sigma * 1000, color = color, marker = 's', linestyle = 'none')
    plt.errorbar(Tcr, sigma * 1000, yerr = dsigma * 1000, ecolor = color, linestyle = 'none', capsize = 2, elinewidth=1, fmt = 'none')
    plt.text(Tcr, sigma * 1000, ser)'''

    dot_depth = abs(Eo - min(E)) # in eV

    if ser not in ['S10', 'S11', 'S12']:
        depth_arr = np.append(depth_arr, dot_depth * 1000) # in meV
    error_dot_depth = dEo * np.sqrt(2) # approximation

    plt.plot(Tcr, Eo, color = color, marker = 's', linestyle = 'none')
    plt.errorbar(Tcr, Eo, yerr = dEo, ecolor = color, linestyle = 'none', capsize = 2, elinewidth=1, fmt = 'none')
    plt.text(Tcr, Eo, ser)
'''
p = np.polyfit(depth_arr, DE_arr, 1)
a1 = p[0]
a2 = p[1]


depth_new = np.linspace(150, 600, 100)
depth_new = np.array(depth_new, dtype = float)
y = a1 * depth_new + a2
#y = a1 * depth_new ** 2 + a2 * depth_new + a3
plt.plot(depth_new, y, 'k--')
txt =  r'y = {:.3f} $\cdot$ x - {:.3f}'.format(a1, abs(a2))
#txt2 =  r'y = {:.3f} $\cdot x^2$ + {:.3f} $\cdot$ x + {:.3f}'.format(a1, a2, a3)
plt.text(400, 250, txt)'''

plt.xlabel(r'$T_{cr}$ [K]')
plt.ylabel(r'$ E_0 $ [eV]')
plt.show()

# outlier = εκτροπη τιμη



