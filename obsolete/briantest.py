from brian2 import *
import numpy as np
import scipy.special  # for the Gamma function

# Parameters
C_m = 0.5*10**(-9) * farad  # Membrane capacitance
g_L = 25*10**(-9) * siemens  # Leak conductance
V_L = -70 * mV  # Leak potential
I = 3 * nA  # Input current
alpha = 0.5  # Fractional order
dt = 0.1 * ms  # Time step
duration = 1000 * ms  # Simulation duration

# Number of time steps
time_steps = int(duration / dt)

# State variables
V = np.zeros(time_steps) * mV # Membrane potential
V[0] = V_L  # Initial condition

# List to store historical voltage for Caputo integration
historical_V = [] * mV

# Simulation loop
for t in range(1, time_steps):
    # Update historical voltage list
    historical_V.append(V[t - 1])
    
    # Calculate the fractional derivative using Caputo definition
    integral_term = sum(historical_V[i] / ((t * dt / second - i * dt / second) ** alpha) 
                    for i in range(len(historical_V)) if (t * dt / second - i * dt / second) > 0)
        fractional_derivative = (integral_term * dt ** (1 - alpha) / gamma(2 - alpha))     
    else:
        fractional_derivative = 0 * mV  # Initial case
    
    # Update the membrane potential
    dV = (-g_L * (V[t - 1] - V_L) + I) / C_m * dt
    V[t] = V[t - 1] + dV + fractional_derivative

# Plotting results
plot(np.arange(time_steps) * dt / ms, V / mV)
xlabel('Time (ms)')
ylabel('Voltage (mV)')
title('Fractional LIF Neuron Dynamics (Caputo Form)')
show()

