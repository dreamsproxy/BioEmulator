import numpy as np

CELL_TYPE = {
    # Passes the processed to all other io channels except the one it got the signal from
    0 : 'PASS',
    # Passes the processed to ALL io channels regardless of which the source is
    1 : 'BLAST',
    # Passes the processed to the opposite side only
    2 : 'OPPOSITE',
    # Passes the processed to only the TOP and BOT faces regardless of source direction
    3 : 'VERTICAL',
    # Passes the processed to the LEFT RIGHT BACK FRONT faces regardless of source direction
    4 : 'HORIZONTAL',
    # Ignores everything EXCEPT: TOP and BOT signals
    5 : 'STRICT_VERTICAL',
    # Ignores TOP and BOT signals
    6 : 'STRICT_HORIZONTAL',
    # If signal came from LEFT RIGHT BACK FRONT, output to TOP and BOT, and vice versa
    7 : 'FLIP_AXIS'
}


class routing:
    def __init__(self):
        pass

    def get_route_types(self):
        return CELL_TYPE