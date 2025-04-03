'''
Shockwave calculator for ideal gas
This is the main file for the shockwave calculator for ideal gas.
It calculates the shockwave properties for multiple mach numbers and plots the entropy ratio.
'''

# Set the path to the parent directory to the global directory
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.insert(0, os.path.abspath(parent_dir))

# Import relevant modules
import numpy as np
import matplotlib.pyplot as plt
import Ideal_gas.shockwave_calculator as sc


def plot_entropy_ratios(fluid, mach_numbers, entropy_ratios):
    """
    Plot the entropy ratios for different Mach numbers.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(mach_numbers, entropy_ratios, marker='o', linestyle='-', color='b')
    plt.title('Entropy Ratios vs. Mach Number')
    plt.xlabel('Mach Number')
    plt.ylabel('Entropy Ratio')
    plt.grid()
    plt.savefig(f'Ideal_gas\Plots\Multiple_shock\entropy_ratios_{fluid}.pdf')
    plt.clf()
    # plt.show()






if __name__ == "__main__":

    # Global variables
    T = 300  # Temperature in K
    p = 101325  # Pressure in Pa
    fluid = 'Helium'  # Fluid type

    Mach_numbers = np.arange(1.1, 5.01, 0.1)  # Mach numbers from 1.1 to 3.0
    entropy_ratios = []  # List to store entropy ratios

    for Mach in Mach_numbers:
        print(f"Calculating for Mach number: {Mach:.2g}")
        # Calculate the shockwave properties for the given Mach number
        shockwave = sc.ShockwaveCalculator(Mach_upstream=Mach, T_upstream=T, p_upstream=p, fluid=fluid)
        entropy_ratios.append(shockwave.entropy_ratio)

    plot_entropy_ratios(fluid, Mach_numbers, entropy_ratios)




