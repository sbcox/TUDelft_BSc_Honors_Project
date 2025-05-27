'''
Shockwave calculator for ideal gas
This file contains the class ShockwaveCalculator, which is used to calculate the shockwave properties of an ideal gas.
Method used as in "The Structure of Shock Waves in the Continuum Theory of Fluids" by D. Gilbarg & D. Paolucci
Created by: Stefan Cox
'''

# Set the path to the parent directory to the global directory
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.insert(0, os.path.abspath(parent_dir))

# Import relevant modules
import numpy as np
import CoolProp.CoolProp as cp
import matplotlib.pyplot as plt


class ShockwaveCalculator:

    def __init__(self, Mach_upstream, p_upstream, T_upstream, fluid, mu_ratio = None, dx = 0.000000001, plot = False):
        '''
        Constructor for the ShockwaveCalculator class
        Inputs:
            Mach: Mach number
            p: pressure in Pa
            T: temperature in K
            dx: step size for integration in m
            fluid: fluid name (e.g. 'Air', 'Water')
        Outputs:
            ShockwaveCalculator object with the following attributes:
                
                Mach: Mach number
                p: pressure in Pa
                T: temperature in K
                fluid: fluid name (e.g. 'Air', 'Water')
                m: mass flow rate in kg/s
                P: total pressure in Pa
                E: energy in J/kg
                alpha: non-dimensional constant
                mu_ref: non-dimensional reference viscosity
                lambda_ref: nondimensional thermal conductivity
        '''
        self.Mach_upstream = Mach_upstream
        self.p_upstream = p_upstream
        self.T_upstream = T_upstream

        self.dx = dx
        self.fluid = fluid

        self.mu_ratio = mu_ratio
        self.mu_upstream = None
        self.lambda_upstream = None

        self.set_consts()
        self.set_downstream_nondims()
        self.set_downstream_eigenvalues()
        self.integrate_shock()
        self.calculate_entropy()
        self.crop_results()

        if plot:
            self.plot_non_dimensional()
            self.plot_dimensional_velocity_cropped()
            self.plot_dimensional_temperature_cropped()
            self.plot_dimensional_pressure_cropped()
            self.plot_entropy_cropped()
            self.plot_nondim_vector()


    def set_mu(self, p, T):
        '''
        Function to set the viscosity of the fluid
        Inputs:
            p: pressure in Pa
            T: temperature in K
        Outputs:
            mu: dynamic viscosity in Pa*s
        '''
        # Check if muratio is set
        if self.mu_ratio is not None and self.mu_upstream is not None:
            # Calculate the dynamic viscosity using the viscosity ratio
            mu = self.mu_upstream * (T/self.T_upstream)**(self.mu_ratio)

            return mu

        # Calculate the dynamic viscosity using CoolProp

        try:
            mu = cp.PropsSI('V', 'P', p, 'T', T, self.fluid)
        except:
            # If CoolProp fails, use a simple model for viscosity
            # This is a simple model and may not be accurate for all fluids
            # print('CoolProp failed to calculate viscosity, using simple model')

            
            if self.fluid == 'Hexamethyldisiloxane':

                # Liquid State
                viscosity_25C = 0.00065 # Pa*s
                mu = np.exp(763.1/T -2.559 + np.log(viscosity_25C))  # https://www.shinetsusilicone-global.com/catalog/pdf/DMF_us.pdf
                return mu


            mu = cp.PropsSI('V', 'P', p, 'T', T, self.fluid)

        return mu
    

    def set_lambda(self, p, T):

        # See if muratio is set
        if self.mu_ratio is not None and self.lambda_upstream is not None:
            # Calculate the thermal conductivity using the viscosity ratio
            lambda_ = self.lambda_upstream * (T/self.T_upstream)**(self.mu_ratio)

            return lambda_


        try:
            lambda_ = cp.PropsSI('L', 'P', p, 'T', T, self.fluid)
        
        except:
            # If CoolProp fails, use a simple model for thermal conductivity
            # This is a simple model and may not be accurate for all fluids
            # print('CoolProp failed to calculate thermal conductivity, using simple model')

            if self.fluid == 'R1233zdE':
            
                # Calculate the critical temperature and density ratios
                Dens = cp.PropsSI('D', 'P', p, 'T', T, self.fluid)
                T_T_c = T / cp.PropsSI('Tcrit', self.fluid)
                D_D_c = Dens / cp.PropsSI('rhocrit', self.fluid)


                # Dilute-gas thermal conductivity
                A_0 = -0.0140033 #W/mK
                A_1 = 0.0378160 #W/mK
                A_2 = -0.00245832 #W/mK

                l_0 = A_0 + A_1 * T_T_c + A_2 * T_T_c**2

                # Residual Thermal conductivity

                # Coefficients for the residual thermal conductivity
                B = np.array([
                    [ 0.862816e-2,  0.914709e-3],   # i = 1
                    [-0.208988e-1, -0.407914e-2],   # i = 2
                    [ 0.511968e-1,  0.845668e-2],   # i = 3
                    [-0.349076e-1, -0.108985e-2],   # i = 4
                    [ 0.975727e-2,  0.538262e-2]    # i = 5
                ])

                l_r = 0.0
                for i in range(5):  # i from 0 to 4 corresponds to 1 to 5
                    term = (B[i][0] + B[i][1] * (T_T_c)) * (D_D_c)**(i + 1)
                    l_r += term

                lambda_ = l_0 + l_r


            elif self.fluid == 'Hexamethyldisiloxane':
                    
                lambda_ = (0.267 * cp.PropsSI('D', 'P', p, 'T', T, self.fluid) - 98.708) / 1000  # W/(m*K) https://pdf.sciencedirectassets.com/272357/1-s2.0-S0021961422X00104/1-s2.0-S0021961422001811/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEE4aCXVzLWVhc3QtMSJHMEUCIB5FVaBLCQFoY0kOO9jgMlLc8n%2FgYL5oihw8hCUVY8iJAiEA17AUjh%2Fvfu02FY6CLwvQh%2BGhM4TY2GmcHvL7LVv7QXcqsgUIFhAFGgwwNTkwMDM1NDY4NjUiDFTXxQv3P7k3nuSbwiqPBbpkI7cwy56LuAgzXU2DBmig5HytTpWqg9Z2JYaXXZNFwo1%2Fw8owYsNhvSIzI6Z3OvxB%2B13XaKV2oIYW2ezbNDL2dgXOhLNZkGhGwZg7Hf2CsdHhsIWZtW35gwnFHnwaqga5LW0CmVtb1zdVNZujHhcSnETUxlmWLgWTHWtwF4zs36EAWxwvR8UU25WN2oEqsWKrTSumaYFR2MdLOqkbMyb2ly%2BDoffV7GGDun2TvcpqPWYRhUXLCd1crUPSHhCgx%2Bcddof%2FxKKJgcjluerUrMLz8skX9g1IVUoJNtD3faGFbpD8v93mrvd7xhSpd0uHRdjt0S%2Fnm2V5KBO19uEza5Q9gqa%2FtJ8AM0S9KjjiOgjICveJuz%2BMb4GzrmFYIp7CP6BEOxQ5d0OawSbYcuHoP8TITWKNmjbkyVVwvDTSgnYJaVHKk0p2Cp6qPVNdkl8q4WHko4BzAA1IBaZZtMQsvX3CDeJRmpFO9sF%2B3DnRp1GaX5DQBe7E90jH978Dq1DoqlVwDyM32I5F7QI%2FBfqnCDxWuaLfpSVvNZkwX1z8jxo42klfjq4x2oq0LrllHsPeHTU1FaXiVWvsET8iDeY%2FlC0vktl3Y8Xfs2S5p3KuDTAQ%2Fo2YJ4hkVIZqVrGEqU6noF7HOtolSlqXsg3ajCQ%2Foh1ECh9tSbAFdJAbKXPgK2pveNkbN1W%2Bc1q2S7jeFopkCWLB%2BrVvPH%2FgVH5nRtU%2BTW%2FoLkdCUqPSResXULRMDWKqZDsQND%2FI2fjAlQV73noqq%2F0o6SSRSGbmAcRl3w40K4IorXx%2FYqEWMD3Y3Hzqo5vm4808TRikeucSNdyBNIYqTsh2kjfR19Jsa6CwDFgJ9%2Bq1hPDf6IdQ6CIwARpOKBAw34jHwQY6sQGUkgqOcy%2FposjiGdKUmMFYPhLJ4%2BtD6EP09Vm3TVRixk4iK%2F9zr%2FnMRrxUe96%2FG0sGCCYDX4ER0pGh63M24hR6%2Fk9YjiYmu7NYEV9QPCGxoqJJz8WdqmLRSUF5AiGi2Hf9Ng1Cb04AccHqGnhndjiMiXOxRX0tdKkKIiEh75b%2FL2mC5f6RQ4aEFq%2Bdn06jWgsS7%2FebxnSJ%2BCOkidaYKHPDn%2BmtP8aDlu0qUHEfsUhxXSc%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20250524T134504Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY5M5WWCEH%2F20250524%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=757dc2634c8d8af15b5661f8d154b39acb3efb049b1cda3bcaa6ede42e5265c9&hash=885399133bf96c118bfcf6d4fa6cbb34771228c58084c2d910223f102dae54dd&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0021961422001811&tid=spdf-a9523620-2b58-4e01-8e96-9e766cea8149&sid=6e72920657c6d443f049314871c850a84364gxrqb&type=client&tsoh=d3d3LXNjaWVuY2VkaXJlY3QtY29tLnR1ZGVsZnQuaWRtLm9jbGMub3Jn&rh=d3d3LXNjaWVuY2VkaXJlY3QtY29tLnR1ZGVsZnQuaWRtLm9jbGMub3Jn&ua=140a5b5156035101500c01&rr=944d3b3caed35811&cc=nl
                return lambda_
                # pass

            else:
                # If the fluid is not in the database, use a simple model for thermal conductivity
                # This is a simple model and may not be accurate for all fluids
                lambda_ = cp.PropsSI('L', 'P', p, 'T', T, self.fluid)


        return lambda_

    def set_consts(self):
        '''
        Function to get the constants for the shockwave calculator
        Inputs: Initial Upstream conditions
            Mach_upstream: upstream Mach number
            p_upstream: upstream pressure in Pa
            T_upstream: upstream temperature in K
            fluid: fluid name (e.g. 'Air', 'Water')
        Outputs:
            # Fluid properties:
            gamma: ratio of specific heats (Cp/Cv)
            R: gas constant in J/(kg*K)
            mu_ref: non-dimensional reference viscosity
            lambda_ref: nondimensional thermal conductivity
            delta: constant based on the specific heat ratio (Cp/Cv)

            # Constants of the flow:
            m: mass flow rate in kg/s
            P: total pressure in Pa
            E: energy in J/kg
            alpha: non-dimensional constant

            # Non-dimensional upstream conditions:
            omega: non-dimensional velocity
            phi: non-dimensional pressure
            theta: non-dimensional temperature

        '''

        
        ''' Get the gas constant and specific heat ratio for the fluid using CoolProp '''
        # Gas constant (J/(kg*K))
        self.R = cp.PropsSI('GAS_CONSTANT', self.fluid) / cp.PropsSI('M', self.fluid)
        # Specific heat ratio (Cp/Cv)
        self.gamma = cp.PropsSI('C', 'P', self.p_upstream, 'T', self.T_upstream, self.fluid) / cp.PropsSI('CVMASS', 'P', self.p_upstream, 'T', self.T_upstream, self.fluid)
        # Constant to ease calculations
        self.delta = (self.gamma - 1)/2
        # Dynamic Viscosity (Pa*s)
        self.mu_upstream = self.set_mu(self.p_upstream, self.T_upstream)
        # Thermal conductivity (W/(m*K))
        self.lambda_upstream = self.set_lambda(self.p_upstream, self.T_upstream)

        ''' Calculate the flow properties at given conditions '''
        # Air Density (kg/m^3)
        rho =  self.p_upstream / (self.R * self.T_upstream)
        # Velocity of flow (m/s)
        u = self.Mach_upstream * np.sqrt(self.gamma * self.R * self.T_upstream)
        
        ''' Constants based on Equations of Mass, momentum, and energy conservation '''
        # Mass flow rate (kg/s)
        self.m = rho * u     
        # Total pressure (Pa)
        self.P = self.p_upstream + self.m * u  
        # Total energy (J/kg)
        self.E = cp.PropsSI('CVMASS', 'P', self.p_upstream, 'T', self.T_upstream, self.fluid) * self.T_upstream  + 0.5 * u**2 + self.p_upstream/rho 

        # Non-dimensional constant alpha
        self.alpha = 2 * self.E * self.m**2 / (self.P**2) - 1 

        # Non-dimensional constant alpha2 (based on NSW eqns)
        self.pressure_ratio = (2 * self.delta + 1) / (self.delta + 1) * self.Mach_upstream**2 - (self.delta) / (self.delta + 1)
        self.alpha2 = self.pressure_ratio / (self.pressure_ratio + 1) ** 2 / (self.delta * (self.delta + 1))

        self.alpha = self.alpha2


        # Nondimensional fluid properties (viscosity and thermal conductivity)
        self.mu_ref_upstream = 4/3 * self.mu_upstream / self.m
        self.lambda_ref_upstream =  self.lambda_upstream / (self.m * cp.PropsSI('CVMASS', 'P', self.p_upstream, 'T', self.T_upstream, self.fluid))

        # Non-dimensional upstream conditions
        # Non-dimensional velocity (omega)
        self.omega_upstream = self.m * u / self.P
        # Non-dimensional pressure (phi)
        self.phi_upstream = self.p_upstream / self.P
        # Non-dimensional temperature (theta)
        self.theta_upstream = self.m**2 * self.R * self.T_upstream / (self.P**2)

        return





    def set_downstream_nondims(self):
        '''
        Function to get the non-dimensional downstream conditions
        Inputs:
            None
        Outputs:
            # Non-dimensional downstream conditions:
            omega_downstream: non-dimensional velocity at downstream condition
            theta_downstream: non-dimensional temperature at downstream condition
        '''

        # Non-dimensional velocity (omega)
        self.omega_downstream = (1/(2*(self.delta + 1)) * (2*self.delta + 1 
                                - np.sqrt(1 - 4*self.delta*(self.delta + 1) * self.alpha)))

        # Non-dimensional temperature (theta)
        self.theta_downstream = (self.delta/(2*(self.delta + 1)**2) * (1 + 2 * (self.delta + 1) * self.alpha
                                + np.sqrt(1 - 4*self.delta*(self.delta + 1) * self.alpha)))
        
        self.phi_downstream = self.theta_downstream / self.omega_downstream

        return
    

    def set_downstream_eigenvalues(self):
        '''
        Function to get the eigenvector at the downstream condition to find init disturbance direction
        Inputs:
            None
        Outputs:
            # Eigenvector which determines direction of initial disturbance:
        '''
        # Derivatives of L and M at downstream condition
        dL_domega = 2 * self.delta * (1 - self.omega_downstream)
        dL_dtheta = 1
        dM_domega = 1 - self.theta_downstream / self.omega_downstream**2
        dM_dtheta = 1 / self.omega_downstream

        # Determine reference thermal conductivity and viscosity at downstream condition
        # Find local temperature
        T_local = self.theta_downstream * self.P **2 / (self.m**2 * self.R)
        p_local = self.phi_downstream * self.P


        mu_ref_local = self.set_mu(p_local, T_local) * 4/3 / self.m
        lambda_ref_local = self.set_lambda(p_local, T_local) / (self.m * cp.PropsSI('CVMASS', 'P', p_local, 'T', T_local, self.fluid))


        # Characteristic matrix for the eigenvalues
        char_matrix = np.array([[dM_domega / mu_ref_local, dM_dtheta / mu_ref_local],
                                [dL_domega / lambda_ref_local, dL_dtheta / lambda_ref_local]])
        
        char_matrix_eigs = np.linalg.eig(char_matrix)

        # Negative eigenvalue is the direction of the initial disturbance
        # Eigenvector is [change in omega, change in theta] for the initial disturbance
        
        # If change in omega is negative, reverse the sign of the eigenvector
        if char_matrix_eigs[0][0] < 0:
            char_matrix_eigs[1][0][0] = -char_matrix_eigs[1][0][0]
            char_matrix_eigs[1][1][0] = -char_matrix_eigs[1][1][0]

        # Define direction of initial disturbance
        self.eigvec_omega = char_matrix_eigs[1][0][0]
        self.eigvec_theta = char_matrix_eigs[1][1][0]

        return



    

    def integrate_shock(self):
        '''
        Function to integrate the shockwave properties
        Inputs:
            None
        Outputs:
            # Non-dimensional values across shockwave:
            omega: non-dimensional velocity array
            theta: non-dimensional temperature array
        '''
        
        # Initialize arrays for non-dimensional values across shockwave (currently reversed)
        self.omega = []
        self.theta = []

        # Initialize arrays for viscosity and thermal conductivity
        self.mu = []
        self.lambda_ = []

        ''' Determine slope of initial disturbance '''
        # Slope of L = 0 curve wrt theta per omega at downstream condition
        dth_dom_L = - 2 * self.delta * (1 - self.omega_downstream)
        # Slope of M = 0 curve in theta per omega at downstream condition
        dth_dom_M = - 2 * self.omega_downstream + 1
        # Average slope of L and M curves at downstream condition (shock must lie between lines)
        dth_dom_avg = (dth_dom_M + dth_dom_L) / 2

        ''' Determine magnitude of initial disturbance '''
        # Relative initial disturbance (arbitrary)
        rel_disturbance = 0.0001

        # Non-dimensional change in shock properties over entire shock
        distance = np.sqrt((self.omega_downstream-self.omega_upstream)**2 + (self.theta_downstream-self.theta_upstream)**2)

        # Initial change in theta and omega based on distance and slope of shock
        init_disturbance = distance * rel_disturbance

        # Initial change in theta and omega based on eigenvalue calculation
        self.set_downstream_eigenvalues()

        self.initial_change_omega = init_disturbance * self.eigvec_omega
        self.initial_change_theta = init_disturbance * self.eigvec_theta



        ''' Add initial values to arrays '''
        self.omega.append(self.omega_downstream + self.initial_change_omega)
        self.theta.append(self.theta_downstream + self.initial_change_theta)


        ''' Integrate shock properties '''
        # Shocked flag to determine if upstream state has been reached
        shocked = False

        # Term to determine shock thickness
        # Maximum change in omega per x
        max_omega_change = 0


        # Loop to integrate shock properties until upstream state is reached
        for i in range(1000000):

            # Calculate L and M at current state
            M = self.omega[-1] + self.theta[-1]/self.omega[-1] - 1
            L = self.theta[-1] - self.delta * ((1-self.omega[-1])**2 + self.alpha)


                

            # Calculate local mu and lambda based on change in conditions
            # Find local temperature
            T_local = self.theta[-1] * self.P **2 / (self.m**2 * self.R)
            p_local = self.theta[-1]/self.omega[-1] * self.P

            mu_local = self.set_mu(p_local, T_local)
            lambda_local = self.set_lambda(p_local, T_local)
            p = 100000  # Pressure in Pa
            if i%1000 == 0:
                
                # print(f'Integrating shock properties: {i/1000000:.2%} done')
                # print(f'Current L: {L:.5g}, Current M: {M:.5g}')
                # print(f'Local mu: {mu_local:.5g}, Local lambda: {lambda_local:.5g}')
                pass

            # Calculate local reference viscosity and thermal conductivity based on change in conditions
            mu_ref_local = self.set_mu(p_local, T_local) * 4/3 / self.m
            lambda_ref_local = self.set_lambda(p_local, T_local) / (self.m * cp.PropsSI('CVMASS', 'P', p_local, 'T', T_local, self.fluid))


            # Append local values to arrays
            self.mu.append(mu_local)
            self.lambda_.append(lambda_local)


            # Calculate change in omega and theta based on L and M
            domega_dx = M / mu_ref_local
            dtheta_dx = L / lambda_ref_local

            # append new values to arrays
            self.omega.append(self.omega[i] - domega_dx * self.dx)
            self.theta.append(self.theta[i] - dtheta_dx * self.dx)

            # Find new maximum change in omega per x
            if max_omega_change < np.abs(M/mu_ref_local):
                max_omega_change = np.abs(M/mu_ref_local)

            # Check if the shock has been reached by ensuring the change in omega and theta is larger than a threshold
            if not shocked and np.abs(self.omega[i] - self.omega[i+1]) > 1e-8 and np.abs(self.theta[i] - self.theta[i+1]) > 1e-8:
                shocked = True

            # Check if final shock state has been reached by ensuring the change in omega and theta is smaller than a threshold
            if np.abs(self.omega[i] - self.omega[i+1]) < 1e-10 and np.abs(self.theta[i] - self.theta[i+1]) < 1e-10 and shocked:
                break

        
        ''' Reverse the arrays to get the correct order '''
        self.omega = np.array(self.omega[::-1])
        self.theta = np.array(self.theta[::-1])
        self.phi = self.theta / self.omega
        self.mu = np.array(self.mu[::-1])
        self.lambda_ = np.array(self.lambda_[::-1])


        ''' Find the dimensional shockwave properties '''
        # Dimensional velocity (m/s)
        self.u = self.omega * self.P / self.m
        # Dimensional pressure (Pa)
        self.p = self.phi * self.P
        # Dimensional temperature (K)
        self.T = self.theta * self.P **2 / (self.m**2 * self.R)
        # Dimensional density (kg/m^3)
        self.rho = self.m / self.u

        
        # Mean free path upstream (m)
        # self.mean_free_path = self.mu[0] / self.p[0] * np.sqrt(np.pi * 1.380649e-23 * self.T_upstream / (2 * cp.PropsSI('M', self.fluid) * 1.66053906660e-27))
        # print('Mean free path:', self.mean_free_path)
        self.mean_free_path = self.mu[0] / self.p[0] * np.sqrt(np.pi * 8.31446261815324 * self.T_upstream / (2 * cp.PropsSI('M', self.fluid)))

        # Shock thickness (-)
        self.thickness = np.abs(self.omega[0]* self.omega[-1] / max_omega_change) / self.mean_free_path
        return
    

    def calculate_entropy(self):
        '''
        Function to calculate the entropy of the shockwave
        Inputs:
            None
        Outputs:
            s_heat: heat entropy (J/(kg*K))
            s_visc: viscous entropy (J/(kg*K))
            s: total entropy (J/(kg*K))
        '''

        # Calculate the entropy of the shockwave
        self.s_heat = np.zeros(len(self.u))
        self.s_visc = np.zeros(len(self.u))
        self.s = np.zeros(len(self.u))

        for i in range(len(self.u)-1):
            self.s_visc[i+1] = self.s_visc[i] + self.mu[i] / self.T[i] * ((self.u[i+1] - self.u[i])/self.dx)**2 * self.dx / self.m
            self.s_heat[i+1] = self.s_heat[i] + self.lambda_[i] / (self.T[i]**2) * ((self.T[i+1] - self.T[i])/self.dx)**2 * self.dx / self.m
            # self.s_heat[i+1] = self.s_heat[i] + self.mu / self.T[i] * ((self.u[i+1] - self.u[i]))**2  * self.m
            # self.s_visc[i+1] = self.s_visc[i] + self.lambda_ / (self.T[i]**2) * ((self.T[i+1] - self.T[i]))**2 *  self.m
 
        self.s = self.s_heat + self.s_visc

        # Calculate entropy ratio
        self.entropy_ratio = self.s_visc[-1] / self.s_heat[-1]
        # print('Entropy ratio:', self.entropy_ratio)
        


        # Calculate reference entropy from Aerodynamics II slides

        T_downstream = self.theta_downstream * self.P **2 / (self.m**2 * self.R)
        p_downstream = self.phi_downstream * self.P

        self.s_ref = (cp.PropsSI('CPMASS', 'P', self.p_upstream, 'T', self.T_upstream, self.fluid) * 
                 np.log(T_downstream/self.T_upstream) -
                 self.R * np.log(p_downstream/self.p_upstream))
        # print('Reference entropy:', self.s_ref)
        # print('Calculated entropy:', self.s[-1])


        

    def plot_non_dimensional(self):
        '''
        Function to plot the non-dimensional shockwave properties
        Inputs:
            None
        Outputs:
            None
        '''

        # Determine bounds of plot
        min_omega = self.omega_downstream + 0.1 * (self.omega_downstream - self.omega_upstream)
        max_omega = self.omega_upstream - 0.1 * (self.omega_downstream - self.omega_upstream)
        min_theta = self.theta_upstream + (self.theta_upstream - 0.25) * 0.1
        max_theta = 0.25 * 1.1

        # Create omega array for plotting
        omega_plot = np.linspace(min_omega, max_omega, 2000)

        # Find L=0 and M=0 lines
        L_0 = np.zeros(len(omega_plot))
        M_0 = np.zeros(len(omega_plot))

        # Calculate L = 0 and M = 0 lines for each omega value
        for j in range(len(omega_plot)):
            L_0[j] = self.delta*((1-omega_plot[j])**2 + self.alpha)
            M_0[j] = omega_plot[j] - omega_plot[j]**2


        # Plot the non-dimensional shockwave properties
        plt.plot(self.omega, self.theta, 'k', label='Shockwave')
        plt.plot(omega_plot, L_0, 'r', label='L=0')
        plt.plot(omega_plot, M_0, 'b', label='M=0')
        plt.plot(self.omega_upstream, self.theta_upstream, 'go', label='Upstream')
        plt.plot(self.omega_downstream, self.theta_downstream, 'yo', label='Downstream')
        # Plot the initial disturbance direction
        plt.quiver(self.omega_downstream, self.theta_downstream, self.eigvec_omega, self.eigvec_theta, angles='xy', scale_units='xy', scale=20, color='y', label='Initial disturbance direction')

        plt.xlabel('Non-dimensional velocity (omega)')
        plt.ylabel('Non-dimensional temperature (theta)')
        plt.title(f'Mach {self.Mach_upstream:.2g} Shockwave Properties of {self.fluid}')
        plt.ylim(min_theta, max_theta)
        plt.xlim(min_omega, max_omega)
        plt.legend()
        plt.grid()
        plt.savefig(f'Ideal_gas\Plots\Single_shock\shockwave_integration_ideal_{self.fluid}_Mach_{int(self.Mach_upstream*10)}.pdf')
        # plt.show()
        plt.clf()


        return
    

    def crop_results(self):
        '''
        Function to crop the results of the shockwave properties
        Inputs:
            Entire shock calculation
        Outputs:
            Cropped results
        '''

        # Find index of value where change in omega is greater than 1e-8 for upstream and downstream
        index_upstream = np.where(np.abs(self.omega - self.omega_upstream) > 1e-3)[0][0]
        index_downstream = np.where(np.abs(self.omega - self.omega_downstream) > 1e-3)[0][-1]

        # print('Index of upstream:', index_upstream)
        # print('Index of downstream:', index_downstream)
        # print('Length of shock:', len(self.omega))

        # Crop the results to only include the shockwave
        self.omega_cropped = self.omega[index_upstream:index_downstream]
        self.theta_cropped = self.theta[index_upstream:index_downstream]
        self.phi_cropped = self.phi[index_upstream:index_downstream]
        self.u_cropped = self.u[index_upstream:index_downstream]
        self.p_cropped = self.p[index_upstream:index_downstream]
        self.T_cropped = self.T[index_upstream:index_downstream]
        self.rho_cropped = self.rho[index_upstream:index_downstream]
        self.s_heat_cropped = self.s_heat[index_upstream:index_downstream]
        self.s_visc_cropped = self.s_visc[index_upstream:index_downstream]
        self.s_cropped = self.s[index_upstream:index_downstream]
        self.mu_cropped = self.mu[index_upstream:index_downstream]
        self.lambda_cropped = self.lambda_[index_upstream:index_downstream]

        return



    

    def plot_dimensional_velocity_cropped(self):
        ''''
        Function to plot the velocity of the shockwave
        Shows the velocity of the shockwave as a function of distance with expected init and final velocities
        '''

        # Distance of shockwave (m)
        x = np.linspace(0, len(self.omega_cropped)*self.dx, len(self.omega_cropped))

        # Upstream and downstream velocities (m/s)
        u_upstream = self.Mach_upstream * np.sqrt(self.gamma * self.R * self.T_upstream)
        u_downstream = self.omega_downstream * self.P / self.m

        # Plot the dimensional velocity
        plt.plot(x, self.u_cropped, 'k', label='Velocity')
        plt.plot(x, u_upstream * np.ones(len(x)), 'r--', label='Upstream Velocity')
        plt.plot(x, u_downstream * np.ones(len(x)), 'g--', label='Downstream Velocity')
        plt.xlabel('Distance (m)')
        plt.ylabel('Velocity (m/s)')
        plt.title(f'Mach {self.Mach_upstream:.2g} Shockwave Velocity of {self.fluid}')
        plt.legend()
        plt.savefig(f'Ideal_gas\Plots\Single_shock\shockwave_velocity_ideal_{self.fluid}_Mach_{int(self.Mach_upstream * 10)}.pdf')
        # plt.show()
        plt.clf()

    def plot_dimensional_temperature_cropped(self):
        ''''
        Function to plot the temperature of the shockwave
        Shows the temperature of the shockwave as a function of distance with expected init and final temperatures
        '''

        # Distance of shockwave (m)
        x = np.linspace(0, len(self.omega_cropped)*self.dx, len(self.omega_cropped))

        # Upstream and downstream temperatures (K)
        T_upstream = self.T_upstream
        T_downstream = self.theta_downstream * self.P **2 / (self.m**2 * self.R)

        # Plot the dimensional temperature
        plt.plot(x, self.T_cropped, 'k', label='Temperature')
        plt.plot(x, T_upstream * np.ones(len(x)), 'r--', label='Upstream Temperature')
        plt.plot(x, T_downstream * np.ones(len(x)), 'g--', label='Downstream Temperature')
        plt.xlabel('Distance (m)')
        plt.ylabel('Temperature (K)')
        plt.title(f'Mach {self.Mach_upstream:.2g} Shockwave Temperature of {self.fluid}')
        plt.legend()
        plt.savefig(f'Ideal_gas\Plots\Single_shock\shockwave_temperature_ideal_{self.fluid}_Mach_{int(10*self.Mach_upstream)}.pdf')
        # plt.show()
        plt.clf()
       


    def plot_dimensional_pressure_cropped(self):
        ''''
        Function to plot the pressure of the shockwave
        Shows the pressure of the shockwave as a function of distance with expected init and final pressures
        '''

        # Distance of shockwave (m)
        x = np.linspace(0, len(self.omega_cropped)*self.dx, len(self.omega_cropped))

        # Upstream and downstream pressures (Pa)
        p_upstream = self.p_upstream
        p_downstream = self.phi_downstream * self.P

        # Plot the dimensional pressure
        plt.plot(x, self.p_cropped, 'k', label='Pressure')
        plt.plot(x, p_upstream * np.ones(len(x)), 'r--', label='Upstream Pressure')
        plt.plot(x, p_downstream * np.ones(len(x)), 'g--', label='Downstream Pressure')
        plt.xlabel('Distance (m)')
        plt.ylabel('Pressure (Pa)')
        plt.title(f'Mach {self.Mach_upstream:.2g} Shockwave Pressure of {self.fluid}')
        plt.legend()
        plt.savefig(f'Ideal_gas\Plots\Single_shock\shockwave_pressure_ideal_{self.fluid}_Mach_{int(10*self.Mach_upstream)}.pdf')
        plt.clf()
        # plt.show()



    def plot_entropy_cropped(self):
        '''
        Function to plot the entropy of the shockwave
        Inputs:
            None
        Outputs:
            None
        '''


        # Distance of shockwave (m)
        x = np.linspace(0, len(self.s_cropped)*self.dx, len(self.s_cropped))

        # Plot the entropy of the shockwave
        plt.plot(x, self.s_heat_cropped, 'r', label='Heat Entropy')
        plt.plot(x, self.s_visc_cropped, 'b', label='Viscous Entropy')
        plt.plot(x, self.s_cropped, 'k', label='Total Entropy')
        plt.plot(x, self.s_ref * np.ones(len(x)), 'g--', label='Reference Entropy')
        
        plt.xlabel('Distance (m)')
        plt.ylabel('Entropy (J/(kg*K))')
        plt.title(f'Mach {self.Mach_upstream:.2g} Shockwave Entropy of {self.fluid}')
        plt.legend()
        plt.savefig(f'Ideal_gas\Plots\Single_shock\shockwave_entropy_ideal_{self.fluid}_Mach_{int(10*self.Mach_upstream)}.pdf')
        plt.clf()
        # plt.show()



    def plot_nondim_vector(self):
        '''
        Function to plot the non-dimensional shockwave properties with the vector field
        '''

                # Determine bounds of plot
        min_omega = self.omega_downstream + 0.1 * (self.omega_downstream - self.omega_upstream)
        max_omega = self.omega_upstream - 0.1 * (self.omega_downstream - self.omega_upstream)
        min_theta = self.theta_upstream + (self.theta_upstream - 0.25) * 0.1
        max_theta = 0.25 * 1.1

        # Create omega array for plotting
        omega_plot = np.linspace(min_omega, max_omega, 2000)

        # Find L=0 and M=0 lines
        L_0 = np.zeros(len(omega_plot))
        M_0 = np.zeros(len(omega_plot))

        # Calculate L = 0 and M = 0 lines for each omega value
        for j in range(len(omega_plot)):
            L_0[j] = self.delta*((1-omega_plot[j])**2 + self.alpha)
            M_0[j] = omega_plot[j] - omega_plot[j]**2

        # Create a grid for the vector field
        omega_grid, theta_grid = np.meshgrid(np.linspace(min_omega, max_omega, 20), np.linspace(min_theta, max_theta, 20))

        # Calculate the M and L values for the vector field
        M_grid = omega_grid + theta_grid / omega_grid - 1
        L_grid = theta_grid - self.delta * ((1 - omega_grid)**2 + self.alpha)

        # Get T and p at the grid points
        T_grid = theta_grid * self.P **2 / (self.m**2 * self.R)
        phi_grid = theta_grid / omega_grid
        p_grid = phi_grid * self.P * omega_grid


        mu_ref_grid = self.set_mu(p_grid, T_grid) * 4/3 / self.m
        lambda_ref_grid = self.set_lambda(p_grid, T_grid) / (self.m)

        for i in range(len(lambda_ref_grid)):
            # Calculate the reference thermal conductivity based on the fluid properties
            lambda_ref_grid[i] /=  cp.PropsSI('CVMASS', 'P', p_grid[i], 'T', T_grid[i], self.fluid)

        # Calculate the derivatives for the vector field
        domega_dx = M_grid / mu_ref_grid
        dtheta_dx = L_grid / lambda_ref_grid

        domega_dx = domega_dx/1e8
        dtheta_dx = dtheta_dx/1e8

        # print(domega_dx)

        # Create a quiver plot for the vector field
        plt.quiver(omega_grid, theta_grid, -domega_dx, -dtheta_dx, angles='xy', scale_units='width', scale=1, color='black', alpha=0.5, label = 'propagation direction')
        
        # Plot the non-dimensional shockwave properties
        plt.plot(self.omega, self.theta, 'k', label='Shockwave')
        plt.plot(omega_plot, L_0, 'r', label='L=0')
        plt.plot(omega_plot, M_0, 'b', label='M=0')
        plt.plot(self.omega_upstream, self.theta_upstream, 'go', label='Upstream')
        plt.plot(self.omega_downstream, self.theta_downstream, 'yo', label='Downstream')
        # Plot the initial disturbance direction
        plt.quiver(self.omega_downstream, self.theta_downstream, self.eigvec_omega, self.eigvec_theta, angles='xy', scale_units='xy', scale=10, color='y', label='Initial disturbance direction')
        plt.xlabel('Non-dimensional velocity (omega)')
        plt.ylabel('Non-dimensional temperature (theta)')
        plt.title(f'Mach {self.Mach_upstream:.2g} Shockwave Properties of {self.fluid}')
        plt.ylim(min_theta, max_theta)
        plt.xlim(min_omega, max_omega)
        plt.legend(fontsize=8)
        plt.grid()
        plt.savefig(f'Ideal_gas\Plots\Single_shock\shockwave_vector_field_ideal_{self.fluid}_Mach_{int(self.Mach_upstream*10)}.pdf')
        plt.clf()


if __name__ == "__main__":
    # Example usage of the ShockwaveCalculator class
    Mach = 2.5
    p = 101325.0
    T = 300.0
    dx = 0.000000001
    fluid = 'air'

    shockwave_calculator = ShockwaveCalculator(Mach, p, T, fluid, dx, plot=True)

    # x = np.linspace(0, len(shockwave_calculator.mu_cropped)*shockwave_calculator.dx, len(shockwave_calculator.mu_cropped))
    # plt.plot(x, shockwave_calculator.mu_cropped, 'k', label='Shockwave')
    # plt.show()

