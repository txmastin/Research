import numpy as np
from flifnetwork import FractionalLIFNetwork

n_inputs = 19
n_outputs = 3
timesteps = 100
dt = 0.002

# All ones scaled up — guaranteed input
dummy_input = np.ones((timesteps, n_inputs)) * 100.0

input_weights = np.eye(n_inputs)
output_weights = np.random.randn(n_inputs, n_outputs) * 0.1

model = FractionalLIFNetwork(
    hidden_layer_size=n_inputs,
    output_layer_size=n_outputs,
    membrane_time_constant=30.0,
    neurons_bias=10.0,          # increased to force spiking
    threshold_voltage=2.5,      # lowered threshold
    reset_voltage=0.0,
    input_weights=input_weights,
    output_weights=output_weights,
    fractional_order=0.85,
    memory_length=20
)
print("Window shape: ", dummy_input.shape)

result = model.simulate_eeg_classification(dummy_input, dt=dt, return_traces=True)

print("Sum of spikes:", result["hidden_spikes"][3])

print("Predicted class:", result["predicted_class"])
print("Output accumulator:", result["output_accumulator"])


