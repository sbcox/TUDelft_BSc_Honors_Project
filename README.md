# TUDelft_BSc_Honors_Project

The aim of this research project is to model the entropy gain due to viscous dissipation (irreversible conversion of mechanical energy to internal energy due to viscous stress) and irreversible heat transfer over a shock wave in XXX fluid. To achieve this, the shock wave will be modeled as a 1-dimensional flow in which the viscous stress and heat transfer are implemented in the continuity, momentum and energy equations. With this model, the entropy production due to both heat transfer and viscous dissipation can be found through numerical integration over the shock, allowing for an analysis of the mechanisms which cause energy loss over the shock wave.


Run global_state.py to attain current project results


Steps in Coding Process:

Completed:
1. Produce Shockwave using normal fluid, ideal gas law, varying viscosity and thermal conductivity with temperature with a simple power law

To do:
2. Introduce more precise method for varying thermal conductivity and viscosity (perhaps coolprop)

3. Introduce non-ideal relations (perhaps vdW, perhaps coolprop, new source found describing possible ways to implement - Gilbarg-ExistenceLimitBehavior)
