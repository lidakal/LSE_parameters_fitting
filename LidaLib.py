import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.linalg import svd
from uncertainties import ufloat

# get_E returns (Sample Code, x, Egap, Eu, Eo from linear aprox., E) of given series 'ser_name' from csv file

def get_E(ser_name):
    data = pd.read_csv('data.csv') # data file contains 22 columns of the form: [Series name, x, 15K, ..., 300K]
    data = data.set_index('Series')

    S = np.array(data.loc[ser_name])
    x = S[0]

    E = S
    E = np.delete(E, 0) # delete x column
    return x, E

# get_Varshni returns the Varshni parameters (a, b) for given x of a series with bowing parameter b

def get_Varshni(x, bowing):
    b, c = bowing
    # Varshni parameters for InN
    a_InN = 4.1E-4 # in eV/K
    b_InN = 454 # in K

    # Varshni parameters for GaN
    a_GaN = 9.14E-4 # in eV/K
    b_GaN = 825 # in K

    # Varshni parameters of series with In fraction x 
    a_v = x * a_InN + (1 - x) * a_GaN - b * x * (1 - x)
    b_v = x * b_InN + (1 - x) * b_GaN - c * x * (1 - x)

    return a_v, b_v    

# f and its derivative, Df, are used to find x for given T numerically

def f(x, kT, sigma, tau, DE):
    return (tau * x) - (((sigma / kT)**2 - x) * np.exp(-x + DE / kT))

def Df(x, kT, sigma, tau, DE):
    return tau + (np.exp(-x + DE / kT)  * ((sigma / kT)**2 - x + 1))

# Newton - Raphson method for roots. Can change the parameters to be varied by adding their values here.

def Newton(f, Df, kT, *params):
    x0 = 105.3
    while(True):
        f1 = f(x0, kT, *params)
        Df1 = Df(x0, kT, *params)
        x1 = x0 - f1/Df1 
        if (abs(x1 - x0) < 1e-6):
            return x1
        x0 = x1

# Calculates x(T). Returns ndarray.

def XofT(kT, sigma, tau, DE):
    result = np.array([])
    for i in kT:
        result = np.append(result, Newton(f, Df, i, sigma, tau, DE))
    return result

# EofT returns E = Eo - ((a * T**2)/(b + T)) - (x(T) * k * T)

def EofT(p, x, T): # p = [[bow], [p0]]
    bowing = p[0]
    p0 = p[1]
    p0 = np.array(p0, dtype = float)
    Eo = p0[0]
    sigma = p0[1]
    tau = p0[2]
    DE = p0[3]
    
    a_v, b_v = get_Varshni(x, bowing)
    k = 8.617333262145e-5 # in eV*K^-1
    kT = k * T
    kTheta = k * b_v
    X = XofT(kT, sigma, tau, DE)
    return Eo - (a_v * np.power(kT,2) / (k * (kTheta + kT))) - (X * kT)

# err_varshni calculates the residuals of EofT for given parameters p, with both varshni parameters fixed

def err_varshni(p, x, T, E): # p = [Eo, sigma, tau, DE]
    bowing = [0, 0]
    pnew = [bowing, p]
    return EofT(pnew, x, T) - E

# err_debye_only calculates the residuals of EofT for given parameters p, only with debye temp. fixed (warning: no global fit for bowing)

def err_debye_only(p, x, T, E): # p = [b, Eo, sigma, tau, DE]
    b = p[0]
    bowing = [b, 0]
    p0 = p[1:]
    pnew = [bowing, p0]
    return EofT(pnew, x, T) - E

# err_varshni_Eo calculates the residuals of EofT for given parameters p, with both varshni parameters and Eo fixed

def err_varshni_Eo(p, x, Eo, T, E): # p = [sigma, tau, DE]
    bowing = [0, 0]
    p0 = np.append(Eo, p)
    pnew = [bowing, p0]
    return EofT(pnew, x, T) - E

# EofT_all_free returns EofT without fixed varshni parameters

def EofT_all_free(p, T): # p = [a_v, b_v, Eo, sigma, tau, DE]
    a_v, b_v, Eo, sigma, tau, DE = p
    k = 8.617333262145e-5 # in eV*K^-1
    kT = k * T
    kTheta = k * b_v
    X = XofT(kT, sigma, tau, DE)
    return Eo - (a_v * np.power(kT,2) / (k * (kTheta + kT))) - (X * kT)


# err_all_free calculates the residuals of EofT_all_free for given parameters p, with all parameters free, without bowing parameter

def err_all_free(p, T, E): # p = [a_v, b_v, Eo, sigma, tau, DE]
    return EofT_all_free(p, T) - E

# find_best_params returns the best (Eo, sigma, tau_ratio, DE) or (b, Eo, sigma, tau_ratio, DE)

def find_best_params(p0, lb, ub, x, T, E, fixed = 'varshni', Eo = -1): # p0 are the initial guesses on the parameters, lb and ub the lower and upper bounds
                                                                # fixed can be 'varshni', 'debye_only', 'varshni_Eo', 'none'
    # make sure arrays are of dtype float
    E = np.array(E, dtype = float)
    p0 = np.array(p0, dtype = float)
    lb = np.array(lb, dtype = float)
    ub = np.array(ub, dtype = float)

    bounds = (lb, ub)

    if fixed == 'varshni':
        fun = err_varshni
        args = (x, T, E)
    elif fixed == 'debye_only':
        fun = err_debye_only
        args = (x, T, E)
    elif fixed == 'varshni_Eo':
        fun = err_varshni_Eo
        args = (x, Eo, T, E)
    elif fixed == 'none':
        fun = err_all_free
        args = (T, E)
    
    res = least_squares(fun, p0, bounds = bounds, args = args)
    popt = res.x

    # Do Moore-Penrose inverse discarding zero singular values.
    _, s, VT = svd(res.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[:s.size]
    pcov = np.dot(VT.T / s**2, VT)
    ysize = len(res.fun)
    cost = 2 * res.cost
    s_sq = cost / (ysize - p0.size)
    pcov = pcov * s_sq
    perr = np.sqrt(np.diag(pcov))

    return popt, perr

# Returns Egap at 0 from Varshni's empirical formula

def get_Egap_at_0(Egap, a, b): # Egap is bandgap at 300 K and a, b are Varshni's parameters
    return Egap + ((a * 300 * 300) / (b + 300))

# global_fit calculates the best parameters for a global bowing 

def global_fit(Series, T, both = False): # if both, p = [b, c, p1, p2, ..., pn]; else p = [b, p1, ..., pn]
    E = np.array([], dtype = float)
    # bowing initial guess
    b0 = -0.002 
    c0 = -0.002
    if both:
        p0_global = np.array([b0, c0], dtype = float)
        lb = np.array([-np.inf, -np.inf], dtype = float)
        ub = np.array([+np.inf, +np.inf], dtype = float)
    else:
        p0_global = np.array([b0], dtype = float)
        lb = np.array([-2.], dtype = float)
        ub = np.array([+2.], dtype = float)

    x = np.array([], dtype = float)

    for ser_name in Series: 
        # Create E = [E1 E2 ... En] (n x 20), x = [x1, x2, ..., xn] (n), p0_global = [b, p1 p2 ... pn] (1 + n x 4)
        x1, E1 = get_E(ser_name)
        E = np.append(E, E1)
        x = np.append(x, x1)
        ini_params = np.array([1.7, 0.03, 0.1, 0.08])
        lb1 = [0, 0, 0, 0]
        ub1 = [3., 1., +np.inf, 1.]
        best_params, errors = find_best_params(ini_params, lb1, ub1, x1, T, E1)
        Eo1 = ufloat(best_params[0], errors[0])
        sigma1 = ufloat(best_params[1], errors[1])
        tau1 = ufloat(best_params[2], errors[2])
        DE1 = ufloat(best_params[3], errors[3])
        p0_global = np.append(p0_global, best_params)
        #lb = np.append(lb, [Eo1.n - Eo1.s, sigma1.n - sigma1.s, tau1.n - tau1.s, DE1.n - DE1.s])
        #ub = np.append(ub, [Eo1.n + Eo1.s, sigma1.n + sigma1.s, tau1.n + tau1.s, DE1.n + DE1.s])

        DElb = DE1.n - 0.4
        if DElb < 0:
            DElb = 0

        taulb = tau1.n - 0.2
        if taulb < 0:
            taulb = 0

        sigmalb = sigma1.n - 0.05
        if sigmalb < 0:
            sigmalb = 0

        lb = np.append(lb, [Eo1.n - 0.1, sigmalb, taulb, DElb])
        ub = np.append(ub, [Eo1.n + 0.1, sigma1.n + 0.05, tau1.n + 0.2, DE1.n + 0.4])

    E = np.array(E, dtype = float)
    p0_global = np.array(p0_global, dtype = float)
    x = np.array(x, dtype = float)
    lb = np.array(lb, dtype = float)
    ub = np.array(ub, dtype = float)

    bounds = (lb, ub)
    #print(bounds)

    res = least_squares(global_err, p0_global, bounds = bounds, args = (x, T, E, both))
    popt = res.x

    # Do Moore-Penrose inverse discarding zero singular values.
    _, s, VT = svd(res.jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[:s.size]
    pcov = np.dot(VT.T / s**2, VT)
    ysize = len(res.fun)
    cost = 2 * res.cost
    s_sq = cost / (ysize - p0_global.size)
    pcov = pcov * s_sq
    perr = np.sqrt(np.diag(pcov))

    return popt, perr


def global_err(p, x, T, E, both = False): # if both, p = [b, c , p1, p2, ..., pn]; else p = [b, p1, ..., pn]
    if both:
        bowing = [p[0], p[1]]
        i = 2
    else:
        bowing = [p[0], 0]
        i = 1

    err = np.array([], dtype = float)

    j = 0

    for x1 in x:
        p0 = [bowing, p[i: i + 4]]
        i = i + 4

        E1 = E[j: j + 20]
        j = j + 20

        y1 = EofT(p0, x1, T)
        err = np.append(err, y1 - E1)
    return err  

