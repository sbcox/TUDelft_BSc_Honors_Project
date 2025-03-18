    

#importing packages

import numpy as np
import CoolProp.CoolProp as cp
import matplotlib.pyplot as plt
import state
import numerical_integration as ni
import entropy_calculation as ec



def plot_shockwave(u, p, T, dx, fluid):
    x = np.linspace(0, len(u) * dx, len(u))

    s_heat, s_visc, s = ec.get_entropy(u, p, T, dx, fluid)

    
    plt.plot(x, T)
    # plt.plot(x, current_state.T * np.ones(len(omega)), 'r--')
    # plt.plot(x, current_state_fin.T * np.ones(len(omega)), 'r--')
    plt.xlabel('x') 
    plt.ylabel('T')
    plt.title('T over x')
    plt.show()

    plt.plot(x, p)
    # plt.plot(x, current_state.p * np.ones(len(omega)), 'r--')
    # plt.plot(x, current_state_fin.p * np.ones(len(omega)), 'r--')
    plt.xlabel('x') 
    plt.ylabel('p')
    plt.title('p over x')
    plt.show()

    plt.plot(x, u)
    # plt.plot(x, current_state.u * np.ones(len(omega)), 'r--')
    # plt.plot(x, current_state_fin.u * np.ones(len(omega)), 'r--')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('u over x')
    plt.show()

    plt.plot(x, s_heat, label='Heat Entropy')
    plt.plot(x, s_visc, label='Viscous Entropy')
    plt.plot(x, s, label='Total Entropy')
    plt.xlabel('x')
    plt.ylabel('Entropy')
    plt.title('Entropy over x')
    plt.legend()
    plt.show()

