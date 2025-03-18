import numpy as np
import CoolProp.CoolProp as cp
import matplotlib.pyplot as plt
import state

'''
Function to calculate the entropy of a shock
'''
def get_entropy(u, p, T, dx, fluid):
    
    s_heat = np.zeros(len(u))
    s_visc = np.zeros(len(u))
    s = np.zeros(len(u))

    for i in range(len(u)-1):
        s_heat[i+1] = s_heat[i] + 4/3 * cp.PropsSI('V', 'P', p[i], 'T', T[i], fluid) / T[i] * ((u[i+1] - u[i])/dx)**2 * dx
        s_visc[i+1] = s_visc[i] + cp.PropsSI('L', 'P', p[i], 'T', T[i], fluid) / (T[i]**2) * ((T[i+1] - T[i])/dx)**2 * dx
        s[i+1] = s_heat[i+1] + s_visc[i+1]

    s_heat /= s[-1]
    s_visc /= s[-1]
    s /= s[-1]

    return s_heat, s_visc, s