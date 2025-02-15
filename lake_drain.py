# -*- coding: utf-8 -*-
"""
"""

import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Set parameters for problem
L = 16.8 #[m], length of pipe
D = 1.22 #[m], diameter of pipe
h0 = 1.52 #[m], initial height of lake
rho = 1000 #[kg/m^3], density of water
g = 9.8 #[m^2/s], gravitational acceleration
eps = 2e-4 #[m], concrete pipe roughness
mu = 0.001 #[Pa*s], viscosity of water
A = 2.02e6 #[m^2], area of lake
tf = 5e5 #[s], time to stop simulation

# simultaneous equations for friction factor and velocity 
def func(x, params):
    [L, D, h, rho, g, eps, mu] = params
    U = x[0]
    fD = x[1]
    Re = rho*U*D/mu
    return [rho*g*h - 0.5*fD*rho*U*U*L/D,
            fD**(-0.5) + 2.0*np.log10((eps/D)/3.7 + 2.51/(Re*fD**0.5))]

def ODEfun(t,y,L, D, rho, g, eps, mu, A, U0, fD0):
    #(L, D, rho, g, eps, mu, A, U0, fD0) = param2

    # solve for velocity U based on current height y
    param = [L,D,y[0],rho,g,eps,mu]
    [U,fD] = fsolve(func,[U0,fD0],param)
    
    # assume constant fD for debugging
    #U = np.sqrt(2*g*y[0]*D/(fD0*L))
    
    # calculate and return derivative    
    return -(np.pi*D*D*U)/(4*A)
    
# def ReStop(t,y,param2):
#     [L, D, rho, g, eps, mu, A, U0, fD0] = param2
#     # calculate the Reynolds number to stop solution once Re<2000 and flow is no longer turbulent
#     # solve for velocity U based on current height y
#     param = (L,D,y[0],rho,g,eps,mu)
#     [U,fD] = fsolve(func,[U0,fD0],param)
#     return (rho*U*D/mu) - 2000
# ReStop.terminal = True
# ReStop.direction = -1

# initial guess and calculation for initial flow conditions
Re0 = 1e7
U0 = Re0*mu/(rho*D)
fD0 = 0.01
params = [L,D,h0,rho,g,eps,mu]
x0 = [U0,fD0]
x = fsolve(func,x0,params)


# solve ODE for height as a function of time with stop condition
param2 = (L,D,rho,g,eps,mu,A,x[0],x[1])
#sol = solve_ivp(ODEfun,[0,tf],[h0],events=ReStop,args=[param2])
sol = solve_ivp(ODEfun,[0,tf],[h0],args=param2,dense_output=True)
t = np.linspace(0,tf,101)
h = sol.sol(t)[0];

# plot height results
plt.figure(figsize=(6.5,6), dpi=200)
#plt.ylim([0,1.5])
plt.plot(t/3600,h*3.281)
plt.xlabel('time (hr)')
plt.ylabel('lake depth (ft)')

# calculate velocity, flow rate, Re, and fD at each height
V = np.zeros(np.shape(t))
V[0] = U0
Q = np.zeros(np.shape(t))
Re = np.zeros(np.shape(t))
fD = np.zeros(np.shape(t))
fD[0] = fD0
param = [L,D,h[0],rho,g,eps,mu]
[V[0],fD[0]] = fsolve(func,[V[0],fD[0]],param)

for i in range(1,len(t)):
    param = [L,D,h[i],rho,g,eps,mu]
    [V[i],fD[i]] = fsolve(func,[V[i-1],fD[i-1]],param)

Q = np.pi*(D/2)**2*V
Re = rho*V*D/mu

# plot Re
plt.figure(figsize=(6.5,6), dpi=200)
plt.plot(t/3600,Re)
plt.yscale("log")
plt.xlabel('time (hr)')
plt.ylabel('pipe Re')
