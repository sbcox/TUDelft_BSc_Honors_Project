#importing packages

import numpy as np
import CoolProp.CoolProp as cp
import matplotlib.pyplot as plt
import entropy_calculation as ec
import state


def find_curve(global_state, omega_f, theta_f, dx):

    omega = []
    theta = []

    upstream = global_state.get_init_state()
    # omega_f = global_state.downstream_2(upstream)
    # theta_f = omega_f - omega_f**2

    dth_dom_L = - 2 * upstream.delta * (1 - omega_f)
    dth_dom_M = - 2 * omega_f + 1
    dth_dom_avg = (dth_dom_M + dth_dom_L) / 2

    # print(dth_dom_L)
    # print(dth_dom_M)
    # print(dth_dom_avg)

    distance = ((upstream.theta - theta_f)**2 + (omega_f - omega_f)**2)**0.5
    init_disturbance = distance * 0.0001

    init_change_theta = dth_dom_avg / ((dth_dom_avg**2 + 1)**0.5) * init_disturbance
    init_change_omega = 1 / ((dth_dom_avg**2 + 1)**0.5) * init_disturbance

    omega.append(omega_f + init_change_omega)
    theta.append(theta_f + init_change_theta)
    
    shocked = False
    for i in range(1000000):
        M = upstream.get_M(omega[i], theta[i])
        L = upstream.get_L(omega[i], theta[i])

        domega_dx = M / global_state.mu_ref
        dtheta_dx = L / global_state.lambda_ref

        omega.append(omega[i] - domega_dx * dx)
        theta.append(theta[i] - dtheta_dx * dx)

        if np.abs(omega[i] - omega[i+1]) > 1e-8 and np.abs(theta[i] - theta[i+1]) > 1e-8:
            shocked = True

        if np.abs(omega[i] - omega[i+1]) < 1e-10 and np.abs(theta[i] - theta[i+1]) < 1e-10 and shocked:
            break
    # print(downstream.omega)
    # print(downstream.theta)
    # print(omega[0])
    # print(theta[0])

    return omega, theta


def get_shock_profile(global_state, dx):

    omega = []
    theta = []

    downstream = global_state.get_init_state()
    omega_f = downstream.omega
    theta_f = downstream.theta

    dth_dom_L = - 2 * downstream.delta * (1 - omega_f)
    dth_dom_M = - 2 * omega_f + 1
    dth_dom_avg = (dth_dom_M + dth_dom_L) / 2

    # print(dth_dom_L)
    # print(dth_dom_M)
    # print(dth_dom_avg)


    init_disturbance = 0.01

    init_change_theta = dth_dom_avg / ((dth_dom_avg**2 + 1)**0.5) * init_disturbance
    init_change_omega = 1 / ((dth_dom_avg**2 + 1)**0.5) * init_disturbance

    omega.append(omega_f)
    theta.append(theta_f)

    omega.append(omega_f + init_change_omega)
    theta.append(theta_f + init_change_theta)
    
    shocked = False
    for i in range(10000):
        if i == 0: continue
        u = omega[i] * global_state.P / global_state.m
        rho = global_state.m / u
        p = theta[i] * global_state.P / omega[i]
        T = cp.PropsSI('T', 'P', p, 'D', rho, global_state.fluid)

        current_state = state.State(u, p, T, global_state.m, global_state.P, global_state.alpha, global_state.fluid)

        M = current_state.get_M(omega[i], theta[i])
        L = current_state.get_L(omega[i], theta[i])

        domega_dx = M / current_state.mu_ref
        dtheta_dx = L / current_state.lambda_ref

        omega.append(omega[i] - domega_dx * dx)
        theta.append(theta[i] - dtheta_dx * dx)

        if np.abs(omega[i] - omega[i+1]) > 1e-8 and np.abs(theta[i] - theta[i+1]) > 1e-8:
            shocked = True

        if np.abs(omega[i] - omega[i+1]) < 1e-10 and np.abs(theta[i] - theta[i+1]) < 1e-10 and shocked:
            break
    # print(downstream.omega)
    # print(downstream.theta)
    # print(omega[0])
    # print(theta[0])


    x = np.linspace(0, len(omega)*dx, len(omega))
    omega = omega[::-1]
    theta = theta[::-1]

    u = np.zeros(len(omega))
    p = np.zeros(len(omega))
    T = np.zeros(len(omega))

    for i in range(len(omega)):
        current = global_state.get_state(omega[i], theta[i])
        u[i] = current.u
        p[i] = current.p
        T[i] = current.T

    s_heat, s_visc, s = ec.get_entropy(u, p, T, dx, global_state.fluid)

    Mach_upstream = u[0] / cp.PropsSI('A', 'T', T[0], 'P', p[0], global_state.fluid)
    Mach_downstream = u[-1] / cp.PropsSI('A', 'T', T[-1], 'P', p[-1], global_state.fluid)

    # create dictionary of all the variables
    output = {
        'omega': omega,
        'theta': theta,
        'x': x,
        'u': u,
        'p': p,
        'T': T,
        's_heat': s_heat,
        's_visc': s_visc,
        's': s,
        'Mach_downstream': Mach_downstream,
        'Mach_upstream': Mach_upstream
    }

    return output



def find_best_start(global_state, dx, theta_i, omega_i):
    return