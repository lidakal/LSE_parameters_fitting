import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from uncertainties import ufloat
import matplotlib.patches as mpl_patches


def E_Varshni(p, T):
    Eo, a, b = p
    res = Eo - (a * T**2) / (b + T) # Eo = 0
    return res

def E_Vina_no_Eb(p, T):
    a, theta = p
    res = - (2 * a) / (np.exp(theta/T) - 1)
    return res

def E_Vina_with_Eb(p, T):
    Eb, a, theta = p
    res = Eb - (a * (1 + (2 / (np.exp(theta/T) - 1))))
    return res

def err_with_Eb(p, T, E):
    Eb, a, theta = p
    y = E_Vina_with_Eb(p, T)
    err = y - E
    #pen = abs(Eb - a) * 10000
    return err #- pen

def err_no_Eb(p, T, E):
    y = E_Vina_no_Eb(p, T)
    err = y - E
    return err

# Functions for fitting

def find_perr(jac, cost, ysize, p0size):
    _, s, VT = np.linalg.svd(jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[:s.size]
    pcov = np.dot(VT.T / s**2, VT)
    cost = 2 * cost
    s_sq = cost / (ysize - p0size)
    pcov = pcov * s_sq
    perr = np.sqrt(np.diag(pcov))
    return perr

def fit(method, p0, T, E, bounds = (-np.inf, +np.inf)):
    if method == 'Vina with Eb':
        fun = err_with_Eb
    elif method == 'Vina no Eb':
        fun = err_no_Eb
    res = least_squares(fun, p0, bounds = bounds, args = (T, E), method = 'lm')
    popt = res.x
    perr = find_perr(res.jac, res.cost, len(E), len(p0))
    return popt, perr

Eo = 0 # in eV

# Varshni parameters for InN
a_InN = 4.1E-4 # in eV/K
b_InN = 454 # in K
p_InN = (Eo, a_InN, b_InN)

# Varshni parameters for GaN
a_GaN = 9.14E-4 # in eV/K
b_GaN = 825 # in K
p_GaN = (Eo, a_GaN, b_GaN)

T = np.array(np.linspace(15, 300, 20), dtype = float)
T_new = np.array(np.linspace(4, 300, 100), dtype = float)

E_GaN = E_Varshni(p_GaN, T)
E_InN = E_Varshni(p_InN, T)

for compound in ['InN']:

    if compound == 'GaN':
        E = E_GaN
    elif compound == 'InN':
        E = E_InN

    # Fit with Eb

    p0 = np.array([1.1, 0.015, 200], dtype = float)

    popt, perr = fit('Vina with Eb', p0, T, E)

    Eb = ufloat(popt[0], perr[0])
    a = ufloat(popt[1], perr[1])
    theta = ufloat(popt[2], perr[2])

    y = E_Vina_with_Eb(popt, T_new)

    s1 = r'$E_B$ = {:.2uP} eV'.format(Eb)
    s2 = r'$a_B$ = {:.2uP} meV'.format(a * 1000)
    s3 = r'$\Theta_B$ = {:.2uP} K'.format(theta)

    handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)] * 4
    labels = ['Fitting parameters:', s1, s2, s3]

    f, ax = plt.subplots(1, 1)
    ax.plot(T, E, color = 'tab:red', marker = 'd', linestyle = 'none')
    ax.plot(T_new, y, 'k--')
    ax.set_xlabel('T in K')
    ax.set_ylabel('E in eV')
    #ax.legend(handles, labels, loc = 'best', fontsize = 'small', fancybox = True, framealpha = 0.7, handlelength = 0, handletextpad = 0)
    #ax.set_title('Varshni Energy Peak Simulation for' + compound + ', Fit for Vina with Eb')
    #plt.savefig('C:/Users/letha/Dropbox/pl/EvsT/Vina/vina_from_varshni/' + compound + '_with_Eb.png')
    plt.show()
    plt.close(f)
'''
    # Without Eb

    p0 = np.array([0.015, 200], dtype = float)

    popt, perr = fit('Vina no Eb', p0, T, E)

    a = ufloat(popt[0], perr[0])
    theta = ufloat(popt[1], perr[1])

    y = E_Vina_no_Eb(popt, T_new)

    s1 = r'$a_B$ = {:.2uP} meV'.format(a * 1000)
    s2 = r'$\Theta_B$ = {:.2uP} K'.format(theta)

    handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)] * 3
    labels = ['Fitting parameters:', s1, s2]

    f, ax = plt.subplots(1, 1)
    ax.plot(T, E, 'b.')
    ax.plot(T_new, y, 'r--')
    ax.set_xlabel('T in K')
    ax.set_ylabel('E in eV')
    ax.legend(handles, labels, loc = 'best', fontsize = 'small', fancybox = True, framealpha = 0.7, handlelength = 0, handletextpad = 0)
    ax.set_title('Varshni Energy Peak Simulation for' + compound + ', Fit for Vina with no Eb')
    #plt.savefig('C:/Users/letha/Dropbox/pl/EvsT/Vina/vina_from_varshni/'+ compound + '_no_Eb.png')
    #plt.show()
    plt.close(f)'''

