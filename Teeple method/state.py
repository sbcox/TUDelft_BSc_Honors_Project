#importing packages

import numpy as np
import CoolProp.CoolProp as cp
import matplotlib.pyplot as plt


'''
the State Class should capture the state of the flow at a given point
it includes functions to swap from physical properties to dimensionless properties
'''

class State:

    def __init__(self, u:float, p:float, T:float, m:float, P:float, alpha:float, fluid='air'):
        self.u = u
        self.p = p
        self.T = T
        self.set_nondims(m, P)
        self.gamma = cp.PropsSI('C', 'P', P, 'T', T, fluid) / cp.PropsSI('CVMASS', 'P', P, 'T', T, fluid)
        self.delta = (self.gamma - 1)/2
        self.alpha = alpha

                # Nondimensional fluid properties (viscosity and thermal conductivity)
        self.mu_ref = 4/3 * cp.PropsSI('V', 'P', p, 'T', T, fluid) / m
        self.lambda_ref =  cp.PropsSI('L', 'P', p, 'T', T, fluid) / (m * cp.PropsSI('CVMASS', 'P', p, 'T', T, fluid))
        
        self.set_M()
        self.set_L()

    
    # def __init__(self, omega:float, phi:float):
    #     self.omega = omega
    #     self.phi = phi

    def set_nondims(self, m, P):
        self.omega = self.get_omega(m, P)
        self.phi = self.get_phi(P)
        self.theta = self.get_theta(m, P)

    def set_dims(self, m, P):
        self.u = self.get_u(m, P)
        self.p = self.get_p(P)

    def get_omega(self, m, P) -> float:
        return m * self.u / P
    
    def get_phi(self, P) -> float:
        return self.p / P
    
    def get_theta(self, m, P) -> float:
        return self.get_omega(m, P) * self.get_phi(P)

    def get_u(self, m, P):
        return self.omega * P / m
    
    def get_p(self, P):
        return self.phi * P
    
    def set_M(self):
        self.M = self.omega + self.theta/self.omega - 1
    
    def set_L(self):
        # print("delta", self.delta)
        self.L = self.theta - self.delta * ((1-self.omega)**2 + self.alpha)

    def get_M(self, omega, theta):
        return omega + theta/omega - 1

    def get_L(self, omega, theta):
        return theta - self.delta * ((1-omega)**2 + self.alpha)
    
    # Get partial derivatives of L and M at a given point (downstream or upstream)

    def get_M_theta(self, omega):
        return 1/omega
    
    def get_L_omega(self, omega):
        return 2 * (1 - omega) * self.delta
    
    def get_M_omega(self, omega, theta):
        return 1 - (theta/(omega**2))
    
    def get_L_theta(self):
        return 1


if __name__ == '__main__':
    state = State(1.5, 100000, 300, 1, 100000, 0.5)
    print(state.omega)
    print(state.phi)
    print(state.theta)
    print(state.u)
    print(state.p)
    print(state.M)
    print(state.L)
    print(state.gamma)
    print(state.delta)
    print(state.alpha)