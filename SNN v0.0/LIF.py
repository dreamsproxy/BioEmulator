import numpy as np
from numba import njit

@njit
def step(params, spike):
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
    
    return wp, new_p

def parse_params(params:dict):
    potential = params['potential']
    dt = params['dt']
    tau = params['tau']
    v_rest = params['v_rest']
    v_reset = params['v_reset']
    v_threshold = params['v_threshold']
    return np.array([potential, dt, tau, v_rest,v_reset, v_threshold])
