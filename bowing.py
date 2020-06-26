import numpy as np
import matplotlib.pyplot as plt
import LidaLib as lil
import pandas as pd
from uncertainties import ufloat
import matplotlib.patches as mpl_patches

T_const = np.array([15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300]) # in K
T_new = np.linspace(15, 300, 100) # Temperature to use for graphs

Series = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12'] # Sample identifiers
#Series = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9']

labels = ['alpha (meV/K)', 'Theta_D (K)', 'bowing(meV/K)', 'Eo (eV)', 'sigma (meV)', 'tau_tr / tau_r', 'Ea - Eo (meV)'] # column labels, 7 in size
results = np.array(labels)

labels_nominal = ['alpha (meV/K)', 'Theta_D (K)', 'bowing(meV/K)', 'Eo (eV)', 'sigma (meV)', 'tau_tr / tau_r', 'Ea - Eo (meV)'] # column labels, 7 in size
results_nominal = np.array(labels_nominal)

#lb = np.append([-1.], np.asarray((+1., 0, 0, 0) * len(Series)))
#ub = np.append([0], np.asarray((3., 1., 1., 1.) * len(Series)))

popt, perr = lil.global_fit(Series, T_const)

b = ufloat(popt[0], perr[0])
print(b)
i = 1
j = 0

for ser_name in Series:
    x1, E1 = lil.get_E(ser_name)
    p0 = popt[i : i + 4]
    errors = perr[i : i + 4]
    i = i + 4
    bowing = [b.n, 0]
    y1 = lil.EofT([bowing, p0], x1, T_new)

    # Text for plot
    
    Eo = ufloat(p0[0], errors[0])

    sigma = ufloat(p0[1], errors[1])

    tau_ratio = ufloat(p0[2], errors[2])

    DE = ufloat(-p0[3], errors[3]) # Ea - Eo

    a_v, b_v = lil.get_Varshni(x1, [b, 0.])

    results = np.vstack((results, [a_v * 1000, b_v, b * 1000, Eo, sigma * 1000, tau_ratio, DE * 1000])) # size must match labels
    results_nominal = np.vstack((results_nominal, [a_v.n * 1000, b_v, b.n * 1000, Eo.n, sigma.n * 1000, tau_ratio.n, DE.n * 1000]))
    
    s1 = r'$\alpha$ = ({:2uP}) meV/K'.format(a_v * 1000)
    s7 = r'$\Theta_D$ = {:.0f} K'.format(b_v)
    s2 = r'$\beta$ = ({:2uP}) meV/K'.format(b * 1000)
    s3 = r'$E_0$ = ({:.2uP}) eV '.format(Eo) 
    s4 = r'$\sigma$ = ({:.2uP}) meV '.format(sigma * 1000)
    s5 = r'$\dfrac{\tau_{tr}}{\tau_r}$ = ' + u'({:.2uP})'.format(tau_ratio) 
    s6 = r'$E_a - E_0$ = ({:.2uP}) meV '.format(DE * 1000)

    handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)] * 10
    labels = ['Fixed parameters:', s7, 'Fitting parameters:', s2, s3, s4, s5, s6, 'Derived parameters:', s1]

    f, ax = plt.subplots(1,1)
    ax.plot(T_const, E1, 'b.', label = 'Data points')
    ax.plot(T_new, y1, 'k--', label = 'Fit')
    ax.set_xlabel('T in K')
    ax.set_ylabel('E in eV')
    ax.legend(handles, labels, loc = 'best', fontsize = 'small', fancybox = True, framealpha = 0.7, handlelength = 0, handletextpad = 0)
    #ax.set_title('Energy Peak of PL vs T for ' + ser_name + ' (' + sample1 + ')')
    #plt.savefig('C:/Users/letha/Dropbox/pl/EvsT/bowing/only_b/const_bound/' + ser_name + '_fit.png', format = 'png')
    #plt.show()

df = pd.DataFrame(data = results)
df.columns = df.iloc[0] # Set the labels as column names in dataframe
df = df.drop(df.index[0]) # Drop the first row of labels, keep only column names 
df = df.set_axis(Series, axis = 'index') # Set the index as the Series indentifiers
df.to_csv('C:/Users/letha/Dropbox/pl/EvsT/bowing/only_b/const_bound/results.csv', index = True) # Save the results in a csv file

df = pd.DataFrame(data = results_nominal)
df.columns = df.iloc[0] # Set the labels as column names in dataframe
df = df.drop(df.index[0]) # Drop the first row of labels, keep only column names 
df = df.set_axis(Series, axis = 'index') # Set the index as the Series indentifiers
#df.to_csv('C:/Users/letha/Dropbox/pl/EvsT/bowing/only_b/const_bound/results_nominal.csv', index = True) # Save the results in a csv file