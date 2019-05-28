
import numpy as np
import matplotlib.pyplot as plt

print('###################### Question 3  ######################')

def runge_kutta(D, D_, time, dt):
    """
    Runge Kuta 4th order method to solve a 2nd order differential equation.
    A second order differential equation is expressed as a system of 2 first order
    linear differential equations.
    Inputs:
    D : initial condition for variable D
    D' : initial condition for the first derivative of variable D
    ts : initial time 
    tf : final time
    dt : step size
    """
    
    f1 = np.zeros(len(time))
    f2 = np.zeros(len(time))
    f1[0] = D  # initial condition f1[0] = D(t=1)
    f2[0] = D_ # initial condition f2[0] = D_(t)
    tc = 1
    

    for i in range(1,len(time)):

        # Runge Kutta formulae
        k0 = dt*f1_( f1[i-1],f2[i-1],tc) 
        l0 = dt*f2_( f1[i-1],f2[i-1],tc) 

        k1 = dt*f1_( f1[i-1]+ 0.5*k0, f2[i-1]+ 0.5*l0, tc+ 0.5*dt) 
        l1 = dt*f2_( f1[i-1]+ 0.5*k0, f2[i-1]+ 0.5*l0, tc+ 0.5*dt) 

        k2 = dt*f1_( f1[i-1]+ 0.5*k1, f2[i-1]+ 0.5*l1, tc+ 0.5*dt) 
        l2 = dt*f2_( f1[i-1]+ 0.5*k1, f2[i-1]+ 0.5*l1, tc+ 0.5*dt) 

        k3 = dt*f1_( f1[i-1]+k2, f2[i-1]+l2, tc+dt) 
        l3 = dt*f2_( f1[i-1]+k2, f2[i-1]+l2, tc+dt) 

        f1[i] = f1[i-1] + (k0 + 2*k1 + 2*k2 + k3)/6. 
        f2[i] = f2[i-1] + (l0 + 2*l1 + 2*l2 + l3)/6. 
           
        tc += dt
        
    return f1

H0 = 7.16e-11 # units yr**-1; Hubble constant 
# start and final time in years, step size of one day
ts, tf, step = 1, 1000, 1/100
time = np.arange(ts, tf, step)
# initial conditions
D0 = [3., 10., 5.]
D0_= [2., -10., 0.]


# scale factor of an Einstein- de Sitter Universe
def a(t):
    return (1.5*H0*t)**(2./3.)

# first derivative of the scale factor of an Einstein- de Sitter Universe a'(t)
def a_(t):
    return ( (H0**(2/3)) * ((1.5*t)**(-1/3)))

# in the density pertubation equation given in equation (4), 
# the following substitutions are made: (in accordance withthe linearized density
# growth equation for an Einstein de Sitter universe.)
# f1 = D; f2= D'; f1_ = f1'= f2
# f2_ = f2' = 1.5*(H0**2/a(t)**3)*f1 - (2*(a_(t)/a(t))*f2)

def f1_(f1,f2,t):
    return f2

def f2_(f1,f2,t):
    return 1.5*((H0**2)*f1/((a(t))**3)) - (2*(a_(t)/a(t))*f2)

# applying the Runge Kutta method to numerically solve the given ODE, 
# for 3 different initial conditions
print("Using RK 4th order method to solve the 2nd order ODE. \n Euler's method can have errors of the order of the step size.")
rk1 = runge_kutta(D0[0], D0_[0], time, step)
rk2 = runge_kutta(D0[1], D0_[1], time, step)
rk3 = runge_kutta(D0[2], D0_[2], time, step)

# analytical solution of the ODE for the 3cases of initial conditions
def rk1_th(t):
    return 3*t**(2/3)
    
def rk2_th(t):
    return 10/t

def rk3_th(t):
    return (3*t**(2/3)) + 2/t

plt.figure(figsize=(10,7))
plt.xscale('log')
plt.yscale('log')
plt.plot((time), (rk1), label='case 1', linewidth='2')
plt.plot(time, rk2, label='case2', linewidth='2')
plt.plot(time, rk3, label='case3', linewidth='2')

plt.plot(time, rk1_th(time), label='case1: Analytical soln', linestyle='dashed')
plt.plot(time, rk2_th(time), label='case2: Analytical soln', linestyle='dashed')
plt.plot(time, rk3_th(time), label='case3: Analytical soln', linestyle='dashed')

plt.title("linear density growth plotted against time for 3cases of initial conditions")
plt.xlabel('log ( time [years] )')
plt.ylabel('log(D): Density growth')
plt.legend()

plt.savefig('RK.png')
