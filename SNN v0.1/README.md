The default neuron type is multipolar, multi in, multi out.

Changes to make:
Implement below neuron rulsets

Multipolar:
    Multi IO:
        N in, N out
    Reaction:
        Weighted sum integration
    Use:
        Learning

Bipolar:
    Single IO:
        1 in, 1 out, directional
    Reaction:
        Selective firing
        Sharp threshold? ReLU? What?
        Test: Use larger decay
    Use:
        Signal Relay
        Used as noise filtering, less sensitie to noise
        Sharp threshold -> less sensitive to small changes
        Sharp threshold -> more sensitive to large changes

Unipolar (Pseudounipolar):
    Bifurcation IO:
        1 in, N out
    Reaction:
        Bulk signalling
        Used for blasting inputs to other neurons fast
        Commonly used for:
            1 network output -> another network input.
            1 cluster output -> another cluster input.

Anaxonic:
    Multi In, No out
    Reaction:
        Inihibit.
    
    Dampens outputs of nearby neurons.

Interneurons:
    Connectors:
        N in, N out
    Reaction:
        Passes signal from dendrites to 
    Use Case:
        
