import numpy as np
import matplotlib.pyplot as plt
import LidaLib as lil
from scipy.optimize import least_squares
from uncertainties import ufloat
import matplotlib.patches as mpl_patches
import pandas as pd

# E(T) with all parameters free

def E_Vina_all_free(T, p):
    a_B, Theta_B, Eo, sigma, tau, DE = p
    k = 8.617333262145e-5 # in eV*K^-1
    kT = k * T
    X = lil.XofT(kT, sigma, tau, DE)
    result = Eo - ((2 * a_B) / (np.exp(Theta_B / T) - 1)) - (X * kT)
    return result

def err_fixed(p, T, E, ab, thetab):
    pnew = np.append([ab, thetab], p)
    y = E_Vina_all_free(T, pnew)
    err = y - E
    return err

def err_free(p, T, E):
    y = E_Vina_all_free(T, p)
    err = y - E
    return err

# Calculate errors
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

def Vina(x):
    a_GaN = 0.058
    theta_GaN = 295.6

    a_InN = 0.03134
    theta_InN = 256.2

    a = (x * a_InN) + ((1 - x) * a_GaN)
    theta = (x * theta_InN) + ((1 - x) * theta_GaN)

    return a, theta


T_const = np.array([15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300]) # in K
T_new = np.linspace(15, 300, 100) # Temperature to use for graphs

#Series = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12'] # Sample identifiers
Series = ['S11']
labels = ['a_B (meV)', 'Theta_B (K)', 'Eo (ev)', 'sigma (meV)', 'tau', 'Ea - Eo (meV)']
results = np.array(labels)

# Now I am gonna fit using different values of a_B, Theta_B for each sample, based on their InN content x

for ser_name in Series:
    x, E = lil.get_E(ser_name)
    E = np.array(E, dtype = float)

    p0_fixed = np.array([1.98, 0.076, 0.00133, 0.15], dtype = float)

    lb_fixed = np.array([0, 0, 0, 0], dtype = float)
    ub_fixed = np.array([3., 1., 2., 1.], dtype = float)

    ab, thetab = Vina(x)

    res = least_squares(err_fixed, p0_fixed, bounds = (lb_fixed, ub_fixed), args = (T_const, E, ab, thetab), verbose = 1)
    popt = res.x
    jac = res.jac
    cost = res.cost
    ysize = len(E)
    psize = len(popt)
    perr = find_perr(jac, cost, ysize, psize)
    print(popt)
    Eo = ufloat(popt[0], perr[0])
    sigma = ufloat(popt[1], perr[1])
    tau = ufloat(popt[2], perr[2])
    DE = ufloat(-popt[3], perr[3])

    results = np.vstack((results, [ab * 1000, thetab, Eo, sigma * 1000, tau, DE * 1000]))

    # text for plot
    s1 = r'$\alpha_B$ = {:.2f} meV'.format(ab * 1000)
    s2 = r'$\Theta_B$ = {:.2f} K'.format(thetab)
    s3 = r'$E_0$ = ({:2uP}) eV'.format(Eo)
    s4 = r'$\sigma$ = ({:2uP}) meV'.format(sigma * 1000)
    s5 = r'$\dfrac{\tau_{tr}}{\tau_r}$ = ' + u'({:.2uP})'.format(tau) 
    s6 = r'$E_a - E_0$ = ({:2uP}) meV'.format(DE * 1000)

    handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)] * 8
    labels = ['Fixed parameters:', s1, s2, 'Fitting parameters:', s3, s4, s5, s6]

    y = E_Vina_all_free(T_new, np.append([ab, thetab], popt))

    f, ax = plt.subplots(1,1)
    ax.plot(T_const, E, 'b.', label = 'Data points')
    ax.plot(T_new, y, 'k--', label = 'Fit')
    ax.set_xlabel('T in K')
    ax.set_ylabel('E in eV')
    ax.legend(handles, labels, loc = 'best', fontsize = 'small', fancybox = True, framealpha = 0.7, handlelength = 0, handletextpad = 0)
    #ax.set_title('Energy Peak of PL vs T for ' + ser_name + ' (' + sample + ')')
    plt.savefig('C:/Users/letha/Dropbox/pl/EvsT/Vina/vina_fixed/from_varshni/' + ser_name + '_fit.png', format = 'png')
    #plt.show()
    plt.close(f)

df = pd.DataFrame(data = results)
df.columns = df.iloc[0] # Set the labels as column names in dataframe
df = df.drop(df.index[0]) # Drop the first row of labels, keep only column names 
df = df.set_axis(Series, axis = 'index') # Set the index as the Series indentifiers
#df.to_csv('C:/Users/letha/Dropbox/pl/EvsT/Vina/vina_fixed/from_varshni/results.csv', index = True) # Save the results in a csv file
