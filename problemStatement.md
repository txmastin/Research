# Problem Statement

What kind of scale-invariant (i.e., scale-free) properties do fractional-order LIF spiking neural networks exhibit? 

- How do they compare to non fractional-order spiking neural networks?

- Are they universal under all fractional-order parameters? e.g., [0-1]

## General overview of methodology:

1.  Identify scale-invariant properties in non fractional-order LIF spiking neural networks


2.  Identify scale-invariant properties in fractional-order LIF spiking neural networks

   - Do scaling exponents change with fractional-order parameter?


3.  Compare scaling exponents between non fractional-order and fractional-order spiking neural networks

- Are scaling exponents the same or similar between non fractional and fractional SNNs?

- Do scaling exponents resemble those found in biological neural networks?
    
- How do they compare to directed percolation / ising models


## Preliminary scale-invariant properties to investigate:

Neuronal avalanches:
        
- Avalanche size distribution 

    - When plotted over log-log plot, is distribution linear -> tau
        
- Avalanche duration distribution

    - When plotted over log-log plot, is distribution linear -> alpha

- Average size, given distribution

    - When plotted over log-log plot, is distribution linear -> gamma

    - Does gamma = (alpha - 1) / (tau - 1)?

## Preliminary models to examine for both fractional- and non fractional-order LIF SNNs:

- Simple feed forward networks
        
    - Fully connected

    - Non-recurrent

- Other feed-foward networks:

    - Sparsely connected

    - Recurrent

    - Convolutional

## Other considerations:
    
- Is the branching ratio near 1? In other words, on average, how many downstream neurons spike for each given neuron 

- How do the networks change pre- to post-training?

    - Do they converge to critical point after training a sufficiently complex task

    - Are scale-free properties of LIF SNNs robust to changes in V_threshold
        
        - V_threshold too low may be subcritical

        - V_threshold too high may be supercritical
