import numpy as np
import matplotlib.pyplot as plt
import LidaLib as lil
import pandas as pd

def Evarshni(T, a, b, p):
    Eo, sigma, tau, DE = p

    k = 8.617333262145e-5 # in eV*K^-1
    kT = k * T
    kTheta = k * b
    X = lil.XofT(kT, sigma, tau, DE)
    return Eo - (a * np.power(kT,2) / (k * (kTheta + kT))) - (X * kT)

def Evina(T, a_B, Theta_B, p):
    Eo, sigma, tau, DE = p
    k = 8.617333262145e-5 # in eV*K^-1
    kT = k * T
    X = lil.XofT(kT, sigma, tau, DE)
    result = Eo - ((2 * a_B)/(np.exp(Theta_B/T)-1)) - (X * kT)
    return result

T_const = np.array([15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300]) # in K
T_new = np.linspace(15, 300, 100) # Temperature to use for graphs


data = pd.read_csv('energies.csv')
data = data.set_index('Series')
#print(data)

res = pd.read_csv('C:/Users/letha/Dropbox/pl/EvsT/bowing/only_b/const_bound/results_nominal.csv') # bowing (B)
#res = pd.read_csv('C:/Users/letha/Dropbox/pl/EvsT/Vina/vina_fixed/from_varshni/with_Eb/res_nominal.csv') # Vina (C)
res = res.set_index('Series')
#print(res)

Series = ['S1', 'S2', 'S3']
#Series = ['S4', 'S5', 'S6']
#Series = ['S7', 'S8', 'S9']
#Series = ['S10', 'S11', 'S12']

syms = [{'color':'tab:blue', 'marker':'d'}, 
        {'color':'tab:green', 'marker': '^'},
        {'color':'tab:red', 'marker':'s'}]

f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex = True)
#f, (ax1, ax2) = plt.subplots(2, 1, sharex = True, gridspec_kw={'height_ratios': [2, 1]})
#f.subplots_adjust(hspace=0.05) # for 2 subplots

#f, ax = plt.subplots(1, 1)

for (ser, sym) in zip(Series, syms):
    E = np.array(data.loc[ser])
    p = np.array(res.loc[ser])  

    # for Vina
    '''
    a, theta, Eo, sigma, tau, DE = p
    a = a / 1000. # in eV/K

    p0 = [Eo, sigma, tau, DE]
    y = Evina(T_new, a, theta, p0)'''

    # for bowing
    
    a, b, bow, Eo, sigma, tau, DE = p
    a = a/1000
    sigma = sigma/1000
    DE = - DE/1000
    p0 = [Eo, sigma, tau, DE] 
    y = Evarshni(T_new, a, b, p0)

    

    if ser == 'S1':
        ax3.plot(T_new, y, 'k--')
        ax3.plot(T_const, E, **sym, linestyle = 'none', label = ser)
        ax3.text(70, 1.87, ser, color = 'tab:blue', weight = 'bold') 
        ax3.set_ylim(1.60, 1.645)
        
    elif ser == 'S2':
        ax2.plot(T_new, y, 'k--')
        ax2.plot(T_const, E, **sym, linestyle = 'none', label = ser)
        ax2.text(220, 1.815, ser, color = 'tab:green', weight = 'bold')
        ax2.set_ylim(1.715, 1.74)
    else:
        ax1.plot(T_new, y, 'k--')
        ax1.plot(T_const, E, **sym, linestyle = 'none', label = ser)
        ax1.text(50, 1.725, ser, color = 'tab:red', weight = 'bold')
        ax1.set_ylim(1.82, 1.865)

# for 3 subplots


# hide the spines between axes
ax3.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)

# hide the ticks between axes
ax1.xaxis.tick_top()
ax2.xaxis.set_ticks_position('none') 
ax3.xaxis.tick_bottom()

d = .01  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the middle axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # top-middle-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # top-middle-right diagonal
ax2.plot((-d, +d), (-d, +d), **kwargs)        # bottom-middle-left diagonal
ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # bottom-middle-right diagonal

kwargs.update(transform=ax3.transAxes)  # switch to the bottom axes
ax3.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax3.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal



# for 2 subplots
'''
# hide the spines between axes
ax2.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)

# hide the ticks between axes
ax1.xaxis.tick_top()
ax2.xaxis.tick_bottom()

d = .01  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
f.text(0.02, 0.5, r'E$_{pk}$ [eV]', va='center', rotation='vertical')'''

ax2.set_ylabel(r'E$_{pk}$ [eV]')
ax3.set_xlabel('T [K]')


#ax1.set_yticks([1.83, 1.87, 1.91, 1.95, 1.99])
plt.show()