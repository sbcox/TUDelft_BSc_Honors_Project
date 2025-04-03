#importing packages

import numpy as np
import CoolProp.CoolProp as cp
import matplotlib.pyplot as plt
import state
import numerical_integration as ni
import entropy_calculation as ec
import plot_shockwave as ps

'''
GlobalState is a class with the aim of defining the global state of the flow
These variables should be constant for the entire flow
'''

class GlobalState:

    m = 0.
    P = 0.
    E = 0.
    alpha = 0.
    mu_ref = 0.
    lambda_ref = 0.

    '''
    Constructor for the GlobalState class
    '''
    def __init__(self, Mach, p, T, fluid='air'):
        # Initial Mach, pressure, and temperature
        self.Mach = Mach
        self.p = p
        self.T = T
        self.fluid = fluid

        # Calculate the global state variables

        
        rho = cp.PropsSI('D', 'P', p, 'T', T, fluid)        # density [kg/m^3]
        u = Mach * cp.PropsSI('A', 'P', p, 'T', T, fluid)   # velocity [m/s]
        
        ''' Constants based on Equations of Mass, momentum, and energy conservation '''
        self.m = rho * u                                    # mass flow rate [kg/s] 
        self.P = p + self.m * u                             # momentum [kg/(m*s^2)] = [Pa]
        self.E = cp.PropsSI('CVMASS', 'P', p, 'T', T, fluid) * T  + 0.5 * u**2 + p/rho # energy [J/kg] = [m^2/s^2]
        self.alpha = 2 * self.E * self.m**2 / (self.P**2) - 1    # non-dimensional constant

        # Nondimensional fluid properties (viscosity and thermal conductivity)
        self.mu_ref = 4/3 * cp.PropsSI('V', 'P', p, 'T', T, fluid) / self.m
        self.lambda_ref =  cp.PropsSI('L', 'P', p, 'T', T, fluid) / (self.m * cp.PropsSI('CVMASS', 'P', p, 'T', T, fluid))

    # Get upstream state
    def get_init_state(self):
        gamma = cp.PropsSI('C', 'P', self.p, 'T', self.T, self.fluid) / cp.PropsSI('CVMASS', 'P', self.p, 'T', self.T, self.fluid)
        u = self.Mach * np.sqrt(self.T * gamma * cp.PropsSI('GAS_CONSTANT', self.fluid) / cp.PropsSI('M', self.fluid))
        p = self.p
        T = self.T
        return state.State(u, p, T, self.m, self.P, self.alpha, self.fluid)


    # Setting downstream values from NSW equations
    # Returns downstream state
    def downstream(self):
        gamma = cp.PropsSI('C', 'P', self.p, 'T', self.T, self.fluid) / cp.PropsSI('CVMASS', 'P', self.p, 'T', self.T, self.fluid)
        Mach_fin = np.sqrt((2 + (gamma - 1) * self.Mach**2) / (2 * gamma * self.Mach**2 - gamma + 1))
        p_fin = self.p * (1 + 2 * gamma * (self.Mach**2 - 1) / (gamma + 1))
        D_fin = cp.PropsSI('D', 'P', self.p, 'T', self.T, self.fluid) * (gamma + 1) * self.Mach**2 / ((gamma - 1) * self.Mach**2 + 2)
        T_fin = cp.PropsSI('T', 'D', D_fin, 'P', p_fin, self.fluid)
        u_fin = Mach_fin * cp.PropsSI('A', 'P', p_fin, 'T', T_fin, self.fluid)
        return state.State(u_fin, p_fin, T_fin, self.m, self.P, self.alpha, self.fluid)
    
    # Finding downstream conditions using Gilbarg's parabolas
    # Returns downstream omega
    def downstream_2(self, init_state):
        a = (init_state.delta + 1)
        b = - 1 - 2 * init_state.delta
        c = init_state.delta * (1 + self.alpha)

        roots = np.roots([a, b, c])
        return min(roots)
    


    # Plotting L and M = 0 lines and vector field of dominant direction in ideal case
    # Input: omega_shock, theta_shock - calculated shock wave curve

    def plot_L_M_bounds(self, omega_shock = [], theta_shock = [], show_M_L = False):  
        # Determine upstream values of omega and theta:
        current_state = self.get_init_state()
        current_state.set_nondims(self.m, self.P)
        theta_0 = current_state.theta
        omega_0 = current_state.omega

        print(current_state.get_L(current_state.omega, current_state.theta))
        print(current_state.get_M(current_state.omega, current_state.theta))

        # Determine downstream values of omega and theta:
        omega_f = global_state.downstream_2(current_state)
        theta_f = omega_f - omega_f**2

        # Determine downstream state
        current_state_fin = self.downstream()
        theta_f = current_state_fin.theta
        omega_f = current_state_fin.omega

        # Create bounds for omega and phi

        omega = np.linspace(omega_f + 0.5 * (omega_f - omega_0), omega_0 - 0.5 * (omega_f - omega_0), 1000)
        theta = np.linspace(theta_0 + 0.5 * (theta_0 - theta_f), theta_f - 0.5 * (theta_0 - theta_f), 1000)

        # Calculate L and M for each omega and theta

        L = np.zeros((len(omega), len(theta)))
        M = np.zeros((len(omega), len(theta)))

        # Partial derivatives of L and M with respect to omega and theta
        L_theta = np.zeros((len(omega), len(theta)))
        M_theta = np.zeros((len(omega), len(theta)))
        L_omega = np.zeros((len(omega), len(theta)))
        M_omega = np.zeros((len(omega), len(theta)))

        # Change in omega and theta per unit distance
        dom_dx = np.zeros((len(omega), len(theta)))
        dth_dx = np.zeros((len(omega), len(theta)))
        

        # Find L and M values for each omega and theta
        for i in range(len(omega)):
            for j in range(len(theta)):
                L[i, j] = current_state_fin.get_L(omega[i], theta[j])
                M[i, j] = omega[i] + theta[j]/omega[i] - 1

                L_theta[i,j] = current_state_fin.get_L_theta()
                M_theta[i,j] = current_state_fin.get_M_theta(omega[i])
                L_omega[i,j] = current_state_fin.get_L_omega(omega[i])
                M_omega[i,j] = current_state_fin.get_M_omega(omega[i], theta[j])

                dom_dx[i, j] = M[i,j] / self.mu_ref 
                dth_dx[i, j] = L[i,j] / self.lambda_ref


        # Find L=0 and M=0 lines
        L_0 = np.zeros(len(omega))
        M_0 = np.zeros(len(omega))

        for j in range(len(omega)):
            L_0[j] = current_state_fin.delta*((1-omega[j])**2 + self.alpha)
            M_0[j] = omega[j] - omega[j]**2

        # Transpose matrices for plotting
        M = M.transpose()
        L = L.transpose()
        dom_dx = dom_dx.transpose()
        dth_dx = dth_dx.transpose()

        # Normalize the dominant direction vector field for plotting
        dom_dx = dom_dx / (dom_dx**2 + dth_dx**2)**0.5
        dth_dx = dth_dx / (dom_dx**2 + dth_dx**2)**0.5


        if show_M_L:
            # Plot M and L over omega and theta
            plt.imshow(L, extent=[omega[0], omega[-1], theta[-1], theta[0]], aspect='auto', vmax = 1, vmin = -1)
            plt.plot(omega, L_0, 'r', label='L = 0')
            plt.ylim([theta[-1], theta[0]])
            plt.xlabel('omega')
            plt.ylabel('theta')
            plt.title('L over omega and theta')
            plt.colorbar()
            plt.legend()
            plt.show()



            plt.imshow(M, extent=[omega[0], omega[-1], theta[-1], theta[0]], aspect='auto', vmax = 1, vmin = -1)
            plt.plot(omega, M_0, 'r', label='M = 0')
            plt.ylim([theta[-1], theta[0]])
            plt.xlabel('omega')
            plt.ylabel('theta')
            plt.title('M over omega and theta')
            plt.colorbar()
            plt.legend()
            plt.show()


        # Find eigenvectors at final state

        eigenvectors = self.find_eigenvalues_Z1()
        eigenvalues = eigenvectors[0]
        eigenvectors = np.array(eigenvectors[1])
        e1x = eigenvectors[0][0] #* eigenvalues[0] / np.abs(eigenvalues[0])
        e1y = eigenvectors[1][0] #* eigenvalues[0] / np.abs(eigenvalues[0])
        e2x = eigenvectors[0][1] #* eigenvalues[1] / np.abs(eigenvalues[1])
        e2y = eigenvectors[1][1] #* eigenvalues[1] / np.abs(eigenvalues[1])

        e1x_scaled = eigenvectors[0][0] / np.abs(omega_f-omega_0) #* eigenvalues[0] / np.abs(eigenvalues[0])
        e1y_scaled = eigenvectors[1][0] / np.abs(theta_f-theta_0) #* eigenvalues[0] / np.abs(eigenvalues[0])
        e2x_scaled = eigenvectors[0][1] / np.abs(omega_f-omega_0) #* eigenvalues[1] / np.abs(eigenvalues[1])
        e2y_scaled = eigenvectors[1][1] / np.abs(theta_f-theta_0) #* eigenvalues[1] / np.abs(eigenvalues[1])

        # print(e1x, e1y)
        # Plot zero lines and eigenvectors at final state
        # plt.imshow(dom_dth, extent=[omega[0], omega[-1], theta[-1], theta[0]], aspect='auto', vmax = 1, vmin = -1)
        # plt.colorbar()

        '''
        Plot vector field of dominant direction, L and M = 0 lines, initial and final states, and calculated shock line (if applicable)
        '''

        plt.quiver(omega[::20], theta[::20], dom_dx[::20, ::20], dth_dx[::20, ::20], color='g', scale=100, label='Dominant direction')
        plt.plot(omega, M_0, label='M')
        plt.plot(omega, L_0, label='L')
        plt.plot(omega_0, theta_0, 'ro', label='Initial state')
        plt.plot(omega_f, theta_f, 'bo', label='Final state')
        plt.plot(omega_shock, theta_shock, 'k', label='Integration Line')
        # plt.quiver(dom_dth, theta, omega, theta, color='g', scale=10, label='Dominant direction')
        plt.quiver(omega_f, theta_f, e1x, e1y, color='r', scale=10, label='Eigenvector 1')
        plt.quiver(omega_f, theta_f, e2x, e2y, color='b', scale=10, label='Eigenvector 2')
        plt.xlabel('omega')
        plt.ylabel('theta')
        plt.ylim([theta[0], theta[-1]])
        plt.title('L and M over omega and theta')
        plt.legend()
        plt.show()

        

    # Find eigenvalues of Z1 matrix at final state
    def find_eigenvalues_Z1(self):
        # Determine upstream bounds of L and M:
        current_state = self.get_init_state()
        current_state.set_nondims(self.m, self.P)
        theta_0 = current_state.theta
        omega_0 = current_state.omega
        phi_0 = current_state.phi

        # Determine downstream bounds of L and M:
        current_state_fin = self.downstream()
        theta_f = current_state_fin.theta
        omega_f = current_state_fin.omega
        phi_f = current_state_fin.phi

        # Define partial derivatives of L and M at a given point (downstream or upstream)

        M_theta = current_state_fin.get_M_theta(omega_f)
        L_theta = current_state_fin.get_L_theta() 
        M_omega = current_state_fin.get_M_omega(omega_f, theta_f)
        L_omega = current_state_fin.get_L_omega(omega_f)

        characteristic_matrix = np.matrix([[M_omega/self.mu_ref, M_theta/self.mu_ref], 
                                           [L_omega/self.lambda_ref, L_theta/self.lambda_ref]])
        
        other_chat_matrix = np.matrix([[M_omega, M_theta],
                                       [L_omega, L_theta]])
        
        # print(characteristic_matrix)

        eigenvectors = np.linalg.eig(other_chat_matrix)
        # print(eigenvectors)
        return eigenvectors
    

    # Get state at a given point
    def get_state(self, omega, theta):
        u = omega * self.P / self.m
        p = self.P * theta/omega
        rho = self.m / u
        T = cp.PropsSI('T', 'D', rho, 'P', p, self.fluid)
        return state.State(u, p, T, self.m, self.P, self.alpha, self.fluid)






if __name__ == '__main__':
    global_state = GlobalState(Mach = 1.5, p = 100000, T = 300, fluid='air')
    # print(global_state.m)
    # print(global_state.P)
    # print(global_state.E)
    # print(global_state.alpha)
    # print(global_state.mu_ref)
    # print(global_state.lambda_ref)

    current_state = global_state.get_init_state()

    current_state_fin = global_state.downstream()


    # Mach_f = current_state_fin.u / cp.PropsSI('A', 'P', current_state_fin.p, 'T', current_state_fin.T, global_state.fluid)
    # p_f = current_state_fin.p
    # T_f = current_state_fin.T

    # global_state_downstream = GlobalState(Mach = Mach_f, p = p_f, T = T_f, fluid='air')

    # omega_f = global_state_downstream.downstream_2(current_state_fin)
    omega_f = current_state_fin.omega
    theta_f = omega_f - omega_f**2

    # p_f = global_state_downstream.P * theta_f / omega_f
    # u_f = omega_f * global_state_downstream.P / global_state_downstream.m
    # T_f = cp.PropsSI('T', 'P', p_f, 'D', global_state_downstream.m / u_f, global_state_downstream.fluid)
    # Mach_f = u_f / cp.PropsSI('A', 'P', p_f, 'T', T_f, global_state_downstream.fluid)

    # global_state_downstream = GlobalState(Mach = Mach_f, p = p_f, T = T_f, fluid='air')

    # global_state.find_eigenvalues_Z1()
    dx = 0.000000001

    # omega, theta = ni.find_best_start(global_state_downstream, dx, theta_f, omega_f)

    shock = ni.get_shock_profile(global_state, dx)

    omega = shock['omega']
    theta = shock['theta']

    global_state.plot_L_M_bounds(omega_shock = omega, theta_shock = theta)
    ps.plot_shockwave(shock['u'], shock['p'], shock['T'], dx, global_state.fluid)
    print(shock['Mach_downstream'])
    print(shock['Mach_upstream'])

    



