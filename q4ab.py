
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import integrate

###################### Question 4  ######################

### 4(a) ###

# Integration using extended mid point rule- 
def intg_extMP(func, limL, limU, n):
    """ 
    Function to integrate using extended midpoint rule
    Inputs-
    func : function to be integrated
    limL, limU : Limits of integration (limL- lower limit, limU - upper limit)
    n : number of steps/divisions in the interval
    """ 
    h = (limU- limL)/n  # size of each interval
    # initializing integral value
    I = 0
    x_mid = limL + (h/2)
    # applying the extended mid-point rule iin the interval:
    for i in range(0, n-1):
        I += func(x_mid)
        x_mid += h
    return I*h  

        
def intg_romb(func, limL, limU, n):
    """ 
    Function to integrate using romberg rule
    Inputs-
    func : function to be integrated
    limL, limU : Limits of integration (limL- lower limit, limU - upper limit)
    n : number of steps/divisions in the interval
    """ 
    S = np.empty((n,n))
    for i in range(0, n):
        # using the trapezoid rule to caculate the integral in intervals that go as 2**i
        S[i, 0] = intg_extMP(func, limL, limU, 2**i)
        for j in range(0, i):
            #if i > n-2: break
            #else: 
            
            # Romberg formula
            S[i, j+1] =( ((4**(j+1))*S[i, j])- S[i-1,j])/ ((4**(j+1))-1)
    
    return S[n-1,n-1]

# function inside integral
def int_func(a1):
    return -1/(a1**3*H(a1)**3)

def H(a1):
    return H0*np.sqrt(omg_m*(a1)**-3 + omg_lam )

# Linear growth factor
def D(a1):
    return (2.5*omg_m*H0**2)*H(a1)*intg_romb(int_func, 0, a1, 10)

def a(z): return 1/(1+z)
def a_(z):
    return H(a(z))*a(z)

omg_m = 0.3
omg_lam = 0.7
H0 = 70 # units: kmps/MPc
z_vals = np.arange(0, 51, 1)
growth_fac = D(a(50))
gfac_th= (2.5*omg_m*H0**2)*H(a(50))*scipy.integrate.quad(int_func, a(50), 0)[0]

# since limits of a are interchanged for the romberg integration, negative of the integral value will give the value required
print('Value of the integral of function under the integral, for z=50 using Romberg integration is',-intg_romb(int_func, 0, a(50), 25))
print('Value of the integral of function under the integral, for z=50 by analytical integration:',3.820791e-10 )
print('Calculatd growth factor at z=50 using Romberg integration is',-growth_fac)
print('Growth factor obtained by analytical integration:', gfac_th)

### 4b ###

## Numerical differenciation
def diff_rich(func, x, m):
    """
    Numerical differenciation using Richardson extrapolation
    func: function to be differentiated
    x : where the function needs to be differentiated at
    """
    d = 2
    h = 0.1
    Diff = np.empty((m,m))
    for i in range(0, m):
        #central differences
        Diff[i, 0] = (func(x+h) - func(x-h))/(2*h) 
        # Richardson extrapolation formula
        for j in range(0,i):
            Diff[i,j+1] =((d**(2*(j+1)))*Diff[i,j]-Diff[i-1,j])/((d**(2*(j+1)))-1)
        # update value of h
        h = h/d
    return Diff[m-1, m-1]

D_t = diff_rich(D, (50), 5)* a_(50)
print('The value of the differential is',D_t)
