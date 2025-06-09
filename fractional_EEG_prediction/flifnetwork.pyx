# cython: boundscheck=False, wraparound=False
import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t ITYPE_t

cdef class FractionalLIFNetwork:
    cdef int hidden_layer_size, output_layer_size
    cdef double membrane_time_constant, threshold_voltage, reset_voltage, neurons_bias
    cdef double fractional_order
    cdef int memory_length
    cdef public np.ndarray hidden_layer_voltages, hidden_layer_spike_states
    cdef np.ndarray constant_hiddens
    cdef public np.ndarray hidden_weights, output_weights
    cdef public np.ndarray gl_coefficients
    cdef public np.ndarray hidden_voltage_history
    cdef public np.ndarray hidden_history_component

    def __init__(self, int hidden_layer_size, int output_layer_size,
                 double membrane_time_constant, double neurons_bias,
                 double threshold_voltage, double reset_voltage,
                 np.ndarray input_weights, np.ndarray output_weights,
                 double fractional_order=0.9, int memory_length=50):

        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size

        self.hidden_layer_voltages = np.zeros(hidden_layer_size, dtype=np.float64)
        self.hidden_layer_spike_states = np.zeros(hidden_layer_size, dtype=np.int32)
        self.constant_hiddens = np.zeros(hidden_layer_size, dtype=np.float64)

        self.hidden_weights = input_weights
        self.output_weights = output_weights

        self.membrane_time_constant = membrane_time_constant
        self.threshold_voltage = threshold_voltage
        self.reset_voltage = reset_voltage
        self.neurons_bias = neurons_bias

        self.fractional_order = fractional_order
        self.memory_length = memory_length

        self.gl_coefficients = self._compute_gl_coefficients(fractional_order, memory_length)
        self.hidden_voltage_history = np.zeros((memory_length, hidden_layer_size), dtype=np.float64)
        self.hidden_history_component = np.zeros((hidden_layer_size), dtype=np.float64)

    def _compute_gl_coefficients(self, double alpha, int length):
        cdef np.ndarray[DTYPE_t, ndim=1] coeffs = np.zeros(length, dtype=np.float64)
        coeffs[0] = -alpha
        cdef int j
        for j in range(1, length):
            coeffs[j] = (1.0 - (alpha + 1.0) / (j+1)) * coeffs[j-1]
        return coeffs

    def reset(self):
        self.hidden_layer_voltages.fill(0)
        self.hidden_voltage_history.fill(0)

    def reset_memory_component(self):
        self.hidden_history_component.fill(0)

    def update(self, np.ndarray[DTYPE_t, ndim=1] inputs, double dt):
        cdef int i, j
        cdef double kernel = dt ** self.fractional_order

        self.hidden_layer_spike_states.fill(0)
        self.reset_memory_component()

        for i in range(self.hidden_layer_size):
            for j in range(self.memory_length):
                self.hidden_history_component[i] += self.gl_coefficients[j] * self.hidden_voltage_history[j,i]

        for i in range(self.hidden_layer_size):
            self.hidden_layer_voltages[i] = (-self.hidden_layer_voltages[i] / self.membrane_time_constant +
                                             self.neurons_bias + inputs[i]) * kernel - self.hidden_history_component[i]
            if self.hidden_layer_voltages[i] >= self.threshold_voltage:
                self.hidden_layer_spike_states[i] = 1
                self.hidden_layer_voltages[i] = self.reset_voltage

        for i in range(self.hidden_layer_size):
            for j in range(self.memory_length - 1)[::-1]:
                self.hidden_voltage_history[j+1,i] = self.hidden_voltage_history[j,i]
            self.hidden_voltage_history[0,i] = self.hidden_layer_voltages[i]

    def simulate_eeg_classification(self, np.ndarray[DTYPE_t, ndim=2] eeg_window,
                                     double dt=0.002, bint return_traces=False):
        cdef int num_steps = eeg_window.shape[0]
        cdef int t, i
        cdef np.ndarray[DTYPE_t, ndim=1] eeg_input
        cdef np.ndarray[DTYPE_t, ndim=1] output_accumulator = np.zeros(self.output_layer_size, dtype=np.float64)

        if return_traces:
            hidden_voltage_traces = np.zeros((num_steps, self.hidden_layer_size), dtype=np.float64)
            hidden_spike_trains = np.zeros((num_steps, self.hidden_layer_size), dtype=np.int32)

        self.reset()

        for t in range(num_steps):
            eeg_input = eeg_window[t]
            self.update(eeg_input, dt)

            # Accumulate output signal as weighted sum of hidden spikes
            for i in range(self.output_layer_size):
                output_accumulator[i] += np.dot(self.hidden_layer_spike_states, self.output_weights[:, i])

            if return_traces:
                for i in range(self.hidden_layer_size):
                    hidden_voltage_traces[t, i] = self.hidden_layer_voltages[i]
                    hidden_spike_trains[t, i] = self.hidden_layer_spike_states[i]

        predicted_class = int(np.argmax(output_accumulator))

        if return_traces:
            return {
                'predicted_class': predicted_class,
                'output_accumulator': output_accumulator,
                'hidden_voltages': hidden_voltage_traces,
                'hidden_spikes': hidden_spike_trains
            }
        else:
            return predicted_class, output_accumulator

