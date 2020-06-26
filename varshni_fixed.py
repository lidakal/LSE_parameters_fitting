import numpy as np
import matplotlib.pyplot as plt
import LidaLib as lil
import pandas as pd
from uncertainties import ufloat
import matplotlib.patches as mpl_patches

T_const = np.array([15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300]) # in K
T_new = np.linspace(15, 300, 100) # Temperature to use for graphs

Series = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12'] # Sample identifiers

labels = ['alpha (meV/K)', 'beta (K)', 'Eo (eV)', 'sigma (meV)', 'tau_tr / tau_r', 'Ea - Eo (meV)'] # column labels, 6 in size
results = np.array(labels)
labels_with_errors = ['alpha (meV/K)', 'beta (K)', 'Eo (eV)', 'dEo (eV)', 'sigma (meV)', 'dsigma (meV)', 
                        'tau_tr / tau_r', 'd(tau_tr / tau_r)', 'Ea - Eo (meV)', 'd(Ea - Eo) (meV)'] # column labels to save with errors 
                                                                                                    # in different columns, 10 in size
results_with_errors = np.array(labels_with_errors)

for ser_name in Series:
    sample, x, Ea, Eu, Eo, E = lil.get_E(ser_name)
    
    # Calculates the best parameters (Eo, sigma, tau, DE) for given series

    p0 = np.array([1.7, 0.03, 10., 0.08])
    lb = [0, 0, 0, 0]
    ub = [3., 1., +np.inf, 1.]
    best_params, errors = lil.find_best_params(p0, lb, ub, x, T_const, E)
    #print(best_params)

    a, b = lil.get_Varshni(x, [0, 0])

    # Text for plot

    Eo = ufloat(best_params[0], errors[0])

    sigma = ufloat(best_params[1], errors[1])

    tau_ratio = ufloat(best_params[2], errors[2])

    DE = ufloat(-best_params[3], errors[3]) # Ea - Eo
    
    s1 = r'$\alpha$ = {:g} meV/K '.format(a * 1000) 
    s2 = r'$\beta$ = {:.0f} K '.format(b) 
    s3 = r'$E_0$ = ({:.2uP}) eV '.format(Eo) 
    s4 = r'$\sigma$ = ({:.2uP}) meV '.format(sigma * 1000)
    s5 = r'$\dfrac{\tau_{tr}}{\tau_r}$ = ' + u'({:.2uP})'.format(tau_ratio) 
    s6 = r'$E_a - E_0$ = ({:.2uP}) meV '.format(DE * 1000)

    handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)] * 8
    labels = ['Fixed parameters:', s1, s2, 'Fitting parameters:', s3, s4, s5, s6]

    f, ax = plt.subplots(1,1)
    ax.plot(T_const, E, 'b.', label = 'Data points')
    y = lil.EofT(np. append([0], best_params), x, T_new)
    ax.plot(T_new, y, 'k--', label = 'Fit')
    ax.set_xlabel('T in K')
    ax.set_ylabel('E in eV')
    ax.legend(handles, labels, loc = 'best', fontsize = 'small', fancybox = True, framealpha = 0.7, handlelength = 0, handletextpad = 0)
    ax.set_title('Energy Peak of PL vs T for ' + ser_name + ' (' + sample + ')')
    #plt.savefig(ser_name + '_fit.png', format = 'png')
    plt.show()
 
    # Results into array

    results = np.vstack((results, [a * 1000, b, Eo, sigma * 1000, tau_ratio, DE * 1000])) # size must match labels
    results_with_errors = np.vstack((results_with_errors, [a * 1000, b, Eo.n, Eo.s, sigma.n * 1000, sigma.s * 1000, 
                                                        tau_ratio.n, tau_ratio.s, DE.n * 1000, DE.s * 1000])) # var.n is the nominal value 
                                                                                                                # and var.s is the st_dev of var

df = pd.DataFrame(data = results)
df.columns = df.iloc[0] # Set the labels as column names in dataframe
df = df.drop(df.index[0]) # Drop the first row of labels, keep only column names 
df = df.set_axis(Series, axis = 'index') # Set the index as the Series indentifiers
#df.to_csv('results.csv', index = True) # Save the results in a csv file

# Same but with errors in different columns

df_new = pd.DataFrame(data = results_with_errors)
df_new.columns = df_new.iloc[0] # Set the labels as column names in dataframe
df_new = df_new.drop(df_new.index[0]) # Drop the first row of labels, keep only column names 
df_new = df_new.set_axis(Series, axis = 'index') # Set the index as the Series indentifiers
#df_new.to_csv('results_with_errors.csv', index = True) # Save the results in a csv file



