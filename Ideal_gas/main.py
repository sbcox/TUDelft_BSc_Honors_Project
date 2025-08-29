'''
Shockwave calculator for ideal gas
This is the main file for the shockwave calculator for ideal gas.
It calculates the shockwave properties for multiple mach numbers and plots the entropy ratio.

How to use:

1. Define initial conditions and fluids which are to be used (lines 204-207)
    Note: ensure that viscosity and thermal conductivity are supported by CoolProp and that fluids are gaseous
2. Choose Mach number range to calculate (line 234)
    Standard: (1.04-2.5)
3. Choose plots to generate (lines 291-294)

4. Run main.py
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
import pickle
import CoolProp.CoolProp as cp


def plot_entropy_ratios(fluid, mach_numbers, entropy_ratios):
    """
    Plot the entropy ratios for different Mach numbers.
    """
    plt.figure(figsize=(4,3.5), layout='constrained')
    plt.plot(mach_numbers, entropy_ratios, marker='o', linestyle='-', color = colors_dict[fluid])
    # plt.title(f'Effect of shock strength on type of entropy generation for {fluid}')
    plt.xlabel('Mach Number')
    plt.ylabel('Entropy due to Viscosity / Entropy due to Heat Conduction')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'Plots\Multiple_shock\entropy_ratios_{fluid}.pdf')
    plt.clf()
    # plt.show()

def plot_entropy_ratios_all(fluid, mach_numbers, entropy_ratios_dict, temp_ratios_dict, colors_dict):
    """
    Plot the entropy ratios and temperature ratios for different Mach numbers and fluids.
    """
    plt.figure(figsize=(4,3.5), layout='constrained')

    fig, ax1 = plt.subplots(figsize=(5,5))

    ax1.set_xlabel('Mach Number')
    ax1.set_ylabel('Entropy due to Viscosity / Entropy due to Heat Conduction')
    for fluid, entropy_ratios in entropy_ratios_dict.items():
        ax1.plot(mach_numbers, entropy_ratios, linestyle='-', color = colors_dict[fluid], label=fluid)

    plt.legend(loc = 'center left')
    ax2 = ax1.twinx() 
    ax2.set_ylabel('Downstream Temperature / Upstream Temperature (dashed)') # instantiate a second Axes that shares the same x-axis
    for fluid, temp_ratios in temp_ratios_dict.items():
        plt.plot(mach_numbers, temp_ratios, linestyle='--', color = colors_dict[fluid], label=fluid)

    ax1.grid()  
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.title('Effect of shock strength on type of entropy generation for different fluids')
    plt.savefig(f'Plots\Multiple_shock\entropy_ratios_all_fluids.pdf')
    plt.clf()
    

def plot_temp_ratios_all(fluid, mach_numbers, temp_ratios_dict):
    """
    Plot the entropy ratios for different Mach numbers and fluids.
    """
    plt.figure(figsize=(4,3.5), layout='constrained')
    for fluid, temp_ratios in temp_ratios_dict.items():
        plt.plot(mach_numbers, temp_ratios, linestyle='-', color = colors_dict[fluid], label=fluid)
    plt.xlabel('Mach Number')
    plt.ylabel('Downstream Temperature / Upstream Temperature')
    plt.grid()
    # plt.title('Effect of shock strength on type of entropy generation for different fluids')
    plt.legend()
    # plt.tight_layout()
    plt.savefig(f'Plots\Multiple_shock\temp_ratios_all_fluids.pdf')
    plt.clf()

def plot_entropy_all(fluid, entropy_totals_dict, mach_numbers, colors_dict):
    """
    Plot the total entropy gain for different Mach numbers and fluids.
    """
    plt.figure(figsize=(4,3.5), layout='constrained')
    for fluid, entropy_total in entropy_totals_dict.items():
        plt.plot(mach_numbers, entropy_total, linestyle='-', color = colors_dict[fluid], label=fluid)
    plt.xlabel('Mach Number')
    plt.ylabel('Total Entropy Gain [J/kg/K]')
    plt.yscale('log')
    plt.grid()
    # plt.title('Effect of shock strength on type of entropy generation for different fluids')
    plt.legend()
    # plt.tight_layout()
    plt.savefig(f'Plots\Multiple_shock\entropy_total_all_fluids.pdf')
    plt.clf()

def plot_entropy_all_comps(fluid, entropy_totals_dict, entropy_totals_heat_dict, entropy_totals_visc_dict, mach_numbers, colors_dict):
    """
    Plot the total entropy gain for different Mach numbers and fluids.
    """
    plt.figure(figsize=(4,3.5), layout='constrained')
    for fluid, entropy_total in entropy_totals_dict.items():
        plt.plot(mach_numbers, entropy_total, linestyle='-', color = colors_dict[fluid], label=fluid)
        plt.plot(mach_numbers, entropy_totals_heat_dict[fluid], linestyle='--', color = colors_dict[fluid], label=f'{fluid} (heat)')
        plt.plot(mach_numbers, entropy_totals_visc_dict[fluid], linestyle=':', color = colors_dict[fluid], label=f'{fluid} (viscous)')
    plt.xlabel('Mach Number')
    plt.ylabel('Total Entropy Gain [J/kg/K]')
    plt.yscale('log')
    plt.grid()
    # plt.title('Effect of shock strength on type of entropy generation for different fluids')
    plt.legend()
    # plt.tight_layout()
    plt.savefig(f'Plots\Multiple_shock\entropy_total_comp_all_fluids.pdf')
    plt.clf()


def plot_shockwave_thicknesses(fluid, mach_numbers, shockwave_thicknesses, shockwave_thicknesses_mfp, colors_dict):
    """
    Plot the shockwave thicknesses for different Mach numbers.
    """

    plt.figure(figsize=(4,3.5), layout='constrained')
    for fluid, shockwave_ts in shockwave_thicknesses.items():
        plt.plot(mach_numbers, shockwave_ts * 1000000, linestyle='-', color = colors_dict[fluid], label=fluid)
    plt.xlabel('Mach Number')
    plt.ylabel('Shock Thickness [μm]')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    # plt.tight_layout()
    plt.savefig(f'Plots\Multiple_shock\shockwave_thicknesses_log.pdf')
    plt.clf()

    plt.figure(figsize=(4,3.5), layout='constrained')
    for fluid, shockwave_ts_mfp in shockwave_thicknesses_mfp.items():
        plt.plot(mach_numbers, shockwave_ts_mfp, linestyle='-', color = colors_dict[fluid], label = fluid)
    plt.xlabel('Mach Number')
    plt.ylabel('Shock Thickness [mean free paths]')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    # plt.tight_layout()
    plt.savefig(f'Plots\Multiple_shock\shockwave_thicknesses_mfp_log.pdf')
    plt.clf()

    plt.figure(figsize=(4,3.5), layout='constrained')
    for fluid, shockwave_ts in shockwave_thicknesses.items():
        plt.plot(mach_numbers, shockwave_ts * 1000000, linestyle='-', color = colors_dict[fluid], label=fluid)
    plt.xlabel('Mach Number')
    plt.ylabel('Shock Thickness [μm]')
    plt.legend()
    plt.grid()
    # plt.tight_layout()
    plt.savefig(f'Plots\Multiple_shock\shockwave_thicknesses.pdf')
    plt.clf()

    plt.figure(figsize=(4,3.5), layout='constrained')
    for fluid, shockwave_thicknesses_mfp in shockwave_thicknesses_mfp.items():
        plt.plot(mach_numbers, shockwave_thicknesses_mfp, linestyle='-', color = colors_dict[fluid], label = fluid)
    plt.xlabel('Mach Number')
    plt.ylabel('Shock Thickness [mean free paths]')
    plt.legend()
    plt.grid()
    # plt.tight_layout()
    plt.savefig(f'Plots\Multiple_shock\shockwave_thicknesses_mfp.pdf')
    plt.clf()


def get_mu_ratio(fluid, use_mu_ratio):
    # Viscosity power - temperature dependency
    if not use_mu_ratio:
        mu_ratio = None
    elif fluid == 'Air':
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
    fluids = ['Argon', 'Air', 'Water', 'CO2', 'R1234zeE', 'Isobutane']  # Fluid type
    colors = [ '#ff7f0e', "#000000", '#1f77b4','#d62728',"#c50194", "#07c507"]  # Colors for plotting
    colors_dict = {fluid: color for fluid, color in zip(fluids, colors)}  # Dictionary for colors
    use_mu_ratio = False  # Use viscosity ratio for calculations
    
    # Create a dictionary to store the entropy ratios for each fluid
    entropy_ratios_dict = {}
    # Create a dictionary to store the shockwave thicknesses for each fluid
    shockwave_thicknesses_dict = {}
    shockwave_thicknesses_dict_mfp = {}  # Shockwave thicknesses in mean free paths
    temperature_ratios_dict = {}  # Temperature ratios for each fluid
    entropy_totals_dict = {}  # Total entropy for each fluid
    entropy_totals_heat_dict = {}  # Total entropy for each fluid (heat)
    entropy_totals_visc_dict = {}  # Total entropy for each fluid (visc)

    # Loop over the fluids and calculate the shockwave properties
    for fluid in fluids:
        # if fluid != 'R1234zeE' and fluid != 'Isobutane':
        #     continue
        if fluid == 'Hexamethyldisiloxane':
            T = 263
        elif fluid == 'Water':
            T = 400

        # Initialize the fluid properties
        mu_ratio = get_mu_ratio(fluid, use_mu_ratio)

        # Define range of Mach numbers to perfom calculations
        Mach_numbers = np.arange(1.04, 2.501, 0.01)  # Mach numbers from 1.1 to 2.5
        entropy_ratios = []  # List to store entropy ratios
        shockwave_thicknesses = []  # List to store shockwave thicknesses
        shockwave_thicknesses_mfp = []  # List to store shockwave thicknesses in mean free paths
        temperature_ratios = []  # List to store temperature ratios
        entropy_total = []  # List to store total entropy
        entropy_total_heat = []  # List to store total entropy (heat)
        entropy_total_visc = []  # List to store total entropy (visc)

        max_worst_entropy = 0

        # loop over the Mach numbers and calculate the entropy ratios
        for Mach in Mach_numbers:
            # print(f"Calculating for Mach number: {Mach} and {fluid}")
            # Calculate the shockwave properties for the given Mach number
            try:
                with open(f'Results\shockwave_results_{fluid}_Mach_{int(Mach*100)}.pkl', 'rb') as f:
                    shockwave = pickle.load(f)
            except FileNotFoundError:
                shockwave = sc.ShockwaveCalculator(Mach_upstream=Mach, T_upstream=T, p_upstream=p, fluid=fluid, mu_ratio=mu_ratio, plot = True)


            # print(f'fluid: {fluid}', 'mu:', shockwave.mu[0], 'lambda:', shockwave.lambda_[0], 'gamma:', 1/(shockwave.gamma-1), 'c_v:', cp.PropsSI('CVMASS', 'T', 400, 'P', 100000, fluid), 'c_p:', cp.PropsSI('CPMASS', 'T', 400, 'P', 100000, fluid), 'R: ', cp.PropsSI('GAS_CONSTANT', 'T', 400, 'P', 100000, fluid))
            # break
            # shockwave.plot_entropy_cropped()
            entropy_total.append(shockwave.s[-1])
            entropy_ratios.append(shockwave.entropy_ratio)
            shockwave_thicknesses.append(shockwave.thickness * shockwave.mean_free_path)
            shockwave_thicknesses_mfp.append(shockwave.thickness)
            temperature_ratios.append((shockwave.T[-1] / shockwave.T[0]))
            entropy_total_heat.append(shockwave.s_heat[-1])
            entropy_total_visc.append(shockwave.s_visc[-1])

            entropy_diff = np.abs((shockwave.s[-1] - shockwave.s_ref)/shockwave.s_ref)

            if entropy_diff > max_worst_entropy:
                max_worst_entropy = entropy_diff

            # print(f"Fluid: {fluid}, Mach: {Mach:.2f}, Entropy Ratio: {shockwave.entropy_ratio:.5g}, Total entropy: {shockwave.s[-1]} Shockwave Thickness: {shockwave.thickness * shockwave.mean_free_path:.5g} m, t_mfp : {shockwave.thickness:.5g} mfp, Temperature Ratio: {shockwave.T[-1] / shockwave.T[0]:.5g}, entropy error: {100*entropy_diff:.5g}%")
            # print(f"Shock thickness: {shockwave.thickness:.5g} mean free paths")
            # print(f"Mean free path: {shockwave.mean_free_path:.5g} m")
            # print(f"Entropy ratio: {shockwave.entropy_ratio:.5g}")
        
        # print(f"Worst entropy error for {fluid}: {100*max_worst_entropy:.5g}%")


        # Convert entropy ratios to numpy array for plotting
        entropy_ratios_dict[fluid] = np.array(entropy_ratios)
        shockwave_thicknesses_dict[fluid] = np.array(shockwave_thicknesses)
        shockwave_thicknesses_dict_mfp[fluid] = np.array(shockwave_thicknesses_mfp)
        temperature_ratios_dict[fluid] = np.array(temperature_ratios)
        entropy_totals_dict[fluid] = np.array(entropy_total)
        entropy_totals_heat_dict[fluid] = np.array(entropy_total_heat)
        entropy_totals_visc_dict[fluid] = np.array(entropy_total_visc)

    # Create plots for all entropy ratios & shockwave thicknesses
    plot_entropy_all(fluid, entropy_totals_dict, Mach_numbers, colors_dict)
    plot_entropy_ratios_all(fluid, Mach_numbers, entropy_ratios_dict, temperature_ratios_dict, colors_dict)
    plot_shockwave_thicknesses(fluid, Mach_numbers, shockwave_thicknesses_dict, shockwave_thicknesses_dict_mfp, colors_dict)
    plot_entropy_all_comps(fluid, entropy_totals_dict, entropy_totals_heat_dict, entropy_totals_visc_dict, Mach_numbers, colors_dict)