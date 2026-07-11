import numpy as np
from numba import njit, prange

@njit()
def step(params, spike):
    params = params.astype(np.float64)
    p, dt, tau, v_rest, v_reset, v_threshold = params
    """
    Simulates a single time step of a spiking neuron model.
    
    Args:
        - 'p' (float): The membrane potential at the previous time step.
        - 'spike' (float): The input current or spike input for the neuron.
        - 'dt' (float): The time step duration.
        - 'tau' (float): Membrane time constant.
        - 'v_rest' (float): Resting potential.
        - 'v_reset' (float): Reset potential after a spike.
        - 'v_threshold' (float): Threshold for firing a spike.
    
    Returns:
        tuple:
            - fire (bool): Whether the neuron fired a spike.
            - new_p (float): The membrane potential after this time step.
    """
    # Update membrane potential using leaky integrate-and-fire equation
    dV = (- (p - v_rest) + spike) * (dt / tau)
    new_p = p + dV
    
    # Check for firing
    fire = new_p >= v_threshold
    wp = new_p
    if fire:
        new_p = v_reset  # Reset the potential if spike occurs
        #print('Fire')
    if wp < np.float64(1e-16) or np.isinf(wp):
        wp = np.float64(1e-16)
    if new_p < np.float64(1e-16) or np.isinf(new_p):
        new_p = np.float64(1e-16)
    return np.float64(wp), np.float64(new_p)

def parse_params(params:dict):
    potential = params['potential']
    dt = params['dt']
    tau = params['tau']
    v_rest = params['v_rest']
    v_reset = params['v_reset']
    v_threshold = params['v_threshold']
    return np.array([potential, dt, tau, v_rest,v_reset, v_threshold])

@njit(parallel=True)
def batch_lif_step(neurons, inputs, thresholds, out_spikes):
    """
    Parallel LIF update across all neurons.
    
    Args:
        neurons: (N, 6) array of neuron parameters.
        inputs: (N,) array of current inputs to neurons.
        thresholds: (N,) array of threshold voltages.
        out_spikes: (N,) output array to be filled with spike magnitudes.
    """
    for i in prange(inputs.shape[0]):
        p, dt, tau, v_rest, v_reset, v_thresh = neurons[i]
        dV = ((v_rest - p) + inputs[i]) * (dt / tau)
        new_p = p + dV

        spike = new_p >= v_thresh[i]
        wp = new_p
        if spike:
            new_p = v_reset

        if wp < 1e-16 or np.isinf(wp):
            wp = 1e-16
        if new_p < 1e-16 or np.isinf(new_p):
            new_p = 1e-16

        out_spikes[i] = wp
        neurons[i, 0] = new_p
