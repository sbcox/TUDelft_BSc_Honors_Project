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
    plt.title(f'Effect of shock strength on type of entropy generation for {fluid}')
    plt.xlabel('Mach Number')
    plt.ylabel('Entropy due to Viscosity / Entropy due to Heat Conduction')
    plt.grid()
    plt.savefig(f'Ideal_gas\Plots\Multiple_shock\entropy_ratios_{fluid}.pdf')
    plt.clf()
    # plt.show()

def plot_entropy_ratios_all(fluid, mach_numbers, entropy_ratios_dict):
    """
    Plot the entropy ratios for different Mach numbers and fluids.
    """
    plt.figure(figsize=(10, 6))
    for fluid, entropy_ratios in entropy_ratios_dict.items():
        plt.plot(mach_numbers, entropy_ratios, linestyle='-', label=fluid)
    plt.xlabel('Mach Number')
    plt.ylabel('Entropy due to Viscosity / Entropy due to Heat Conduction')
    plt.grid()
    plt.title('Effect of shock strength on type of entropy generation for different fluids')
    plt.legend()
    plt.savefig(f'Ideal_gas\Plots\Multiple_shock\entropy_ratios_all_fluids.pdf')
    plt.clf()

def plot_shockwave_thicknesses(fluid, mach_numbers, shockwave_thicknesses, shockwave_thicknesses_mfp):
    """
    Plot the shockwave thicknesses for different Mach numbers.
    """

    plt.figure(figsize=(10, 6))
    for fluid, shockwave_thicknesses in shockwave_thicknesses.items():
        plt.plot(mach_numbers, shockwave_thicknesses * 1000000, linestyle='-', label=fluid)
    plt.xlabel('Mach Number')
    plt.ylabel('Shockwave Thickness [μm]')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.savefig(f'Ideal_gas\Plots\Multiple_shock\shockwave_thicknesses.pdf')
    plt.clf()

    plt.figure(figsize=(10, 6))
    for fluid, shockwave_thicknesses_mfp in shockwave_thicknesses_mfp.items():
        plt.plot(mach_numbers, shockwave_thicknesses_mfp, linestyle='-', label = fluid)
    plt.xlabel('Mach Number')
    plt.ylabel('Shockwave Thickness [mean free paths]')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.savefig(f'Ideal_gas\Plots\Multiple_shock\shockwave_thicknesses_mfp.pdf')
    plt.clf()


def get_mu_ratio(fluid, use_mu_ratio):
    # Viscosity power - temperature dependency
    if not use_mu_ratio:
        mu_ratio = None
    elif fluid == 'air':
        mu_ratio = 0.768
    elif fluid == 'Helium':
        mu_ratio = 0.647
    elif fluid == 'Argon':
        mu_ratio = 0.816
    else:
        mu_ratio = None

    return mu_ratio



### LOOK AT THIS
#https://pubs-acs-org.tudelft.idm.oclc.org/doi/full/10.1021/acs.iecr.1c02154


if __name__ == "__main__":

    # Global variables
    T = 300  # Temperature in K
    p = 100000  # Pressure in Pa
    fluids = ['air', 'Argon', 'R1233zdE', 'Hexamethyldisiloxane']  # Fluid type
    use_mu_ratio = False  # Use viscosity ratio for calculations

    # Create a dictionary to store the entropy ratios for each fluid
    entropy_ratios_dict = {}
    # Create a dictionary to store the shockwave thicknesses for each fluid
    shockwave_thicknesses_dict = {}
    shockwave_thicknesses_dict_mfp = {}  # Shockwave thicknesses in mean free paths

    # Loop over the fluids and calculate the shockwave properties
    for fluid in fluids:
        if fluid == 'Hexamethyldisiloxane':
            T = 263
        # Initialize the fluid properties
        mu_ratio = get_mu_ratio(fluid, use_mu_ratio)

        # Define range of Mach numbers to perfom calculations
        Mach_numbers = np.arange(1.04, 2.501, 0.01)  # Mach numbers from 1.1 to 3.0
        entropy_ratios = []  # List to store entropy ratios
        shockwave_thicknesses = []  # List to store shockwave thicknesses
        shockwave_thicknesses_mfp = []  # List to store shockwave thicknesses in mean free paths

        # loop over the Mach numbers and calculate the entropy ratios
        for Mach in Mach_numbers:
            print(f"Calculating for Mach number: {Mach:.2g}")
            # Calculate the shockwave properties for the given Mach number
            shockwave = sc.ShockwaveCalculator(Mach_upstream=Mach, T_upstream=T, p_upstream=p, fluid=fluid, mu_ratio=mu_ratio, plot = False)
            entropy_ratios.append(shockwave.entropy_ratio)
            shockwave_thicknesses.append(shockwave.thickness * shockwave.mean_free_path)
            shockwave_thicknesses_mfp.append(shockwave.thickness)
            print(f"Shock thickness: {shockwave.thickness:.5g} mean free paths")
            print(f"Mean free path: {shockwave.mean_free_path:.5g} m")
            print(f"Entropy ratio: {shockwave.entropy_ratio:.5g}")


        # Convert entropy ratios to numpy array for plotting
        entropy_ratios_dict[fluid] = np.array(entropy_ratios)
        shockwave_thicknesses_dict[fluid] = np.array(shockwave_thicknesses)
        shockwave_thicknesses_dict_mfp[fluid] = np.array(shockwave_thicknesses_mfp)



    # Create plots for all entropy ratios & shockwave thicknesses
    plot_entropy_ratios_all(fluid, Mach_numbers, entropy_ratios_dict)
    plot_shockwave_thicknesses(fluid, Mach_numbers, shockwave_thicknesses_dict, shockwave_thicknesses_dict_mfp)
