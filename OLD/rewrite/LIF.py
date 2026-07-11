import numpy as np
from numba import njit
from collections import OrderedDict

def default_params() -> OrderedDict:
    params = OrderedDict({
        'p'           : np.float32(-65.0),
        'v_rest'      : np.float32(-65.0),
        'dt'          : np.float32(0.100),
        'v_reset'     : np.float32(-65.0),
        'v_threshold' : np.float32(-52.0),
        'tau'         : np.float32(100.0),
        'tau_input'   : np.float32(2.000),
        'refrac'      : int(0)
    })
    return params

@njit
def step(params, I_syn, signal, refrac_counter):
    p, dt, tau, tau_input, v_rest, v_reset, v_threshold, refrac_time = params
    """
    Simulates a single time step of a spiking neuron model with synaptic current decay and refractory period.

    Args:
        - 'p' (float): Membrane potential at the previous time step.
        - 'I_syn' (float): Synaptic input current at the previous time step.
        - 'signal' (float): New external input.
        - 'refrac_counter' (int): Time steps remaining in refractory period.
        - 'dt' (float): Time step duration.
        - 'tau' (float): Membrane time constant.
        - 'tau_input' (float): Synaptic input current decay time constant.
        - 'v_rest' (float): Resting potential.
        - 'v_reset' (float): Reset potential after a spike.
        - 'v_threshold' (float): Threshold for firing a spike.
        - 'refrac_time' (int): Duration of refractory period.

    Returns:
        tuple:
            - fire (bool): Whether the neuron fired a spike.
            - new_p (float): Updated membrane potential.
            - new_I_syn (float): Updated synaptic input current.
            - new_refrac_counter (int): Updated refractory counter.
    """

    # Update synaptic current (exponential decay + new input)
    I_syn = I_syn * np.exp(-dt / tau_input) + signal  

    # If in refractory period, force membrane potential to reset value and decrement counter
    if refrac_counter > 0:
        new_p = v_reset
        new_refrac_counter = refrac_counter - 1
        fire = False  # Neuron cannot fire during absolute refractory period
    else:
        # Update membrane potential
        dV = (- (p - v_rest) + I_syn) * (dt / tau)
        new_p = p + dV

        # Check for firing
        fire = new_p >= v_threshold
        if fire:
            new_p = v_reset  # Reset membrane potential
            new_refrac_counter = refrac_time  # Enter refractory period
        else:
            new_refrac_counter = 0  # No refractory period if no spike

    return new_p, I_syn, fire, new_refrac_counter
