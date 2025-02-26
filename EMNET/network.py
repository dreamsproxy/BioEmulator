import numpy as np
import cv2
CELL_JOBS = {
    0 : 'SUM',
    1 : 'AVG',
    2 : 'MIN',
    3 : 'MAX',
    4 : 'STD',
}

CELL_SIGNAL_TYPE = {
    0 : 'REPEAT',
    1 : 'DISTRIBUTE',
}

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

class Cell:
    def __init__(self, cell_id, cell_type:int, cell_job:int, cell_signal_type:int):
        if cell_job not in list(CELL_JOBS.keys()):
            raise NotImplementedError(f'The provided cell job does not exist!\n\n\tAllowed jobs are:\n\t{CELL_JOBS}')
        if cell_signal_type not in list(CELL_SIGNAL_TYPE.keys()):
            raise NotImplementedError(f'The provided cell signal type does not exist!\n\n\tAllowed jobs are:\n\t{CELL_SIGNAL_TYPE}')
        if cell_type not in list(CELL_TYPE.keys()):
            raise NotImplementedError(f'The provided cell type does not exist!\n\n\tAllowed jobs are:\n\t{CELL_TYPE}')

        self.cell_id = cell_id
        self.cell_type = cell_type
        self.cell_job = cell_job
        self.cell_signal_type = cell_signal_type
        self.directions = {
            0 : 'TOP',
            1 : 'BOT',
            2 : 'LEFT',
            3 : 'RIGHT',
            4 : 'FRONT',
            5 : 'BACK'
        }
        if self.cell_type == 5:
            self.n_neighbors = 2
        if self.cell_type == 6:
            self.n_neighbors = 4
        else:
            self.n_neighbors = 6


    def _func(self, inputs:np.ndarray):
        if self.cell_job == 0:
            return inputs.sum()
        elif self.cell_job == 1:
            return inputs.mean()
        elif self.cell_job == 2:
            return inputs.min()
        elif self.cell_job == 3:
            return inputs.max()
        elif self.cell_job == 4:
            return inputs.std()

    def step(self, inputs:np.ndarray):
        if self.cell_type == 5:
            inputs[2:] = np.float64(0.0)
        if self.cell_type == 6:
            inputs[0:2] = np.float64(0.0)
        print(inputs)
        y = self._func(inputs)

        if self.cell_signal_type == 0:
            y = [y for i in range(6)]
        elif self.cell_signal_type == 1:
            y = [y / self.n_neighbors for i in range(6)]
        if self.cell_signal_type == 0:
            return y
        y[]

c = Cell('0', 6, 4, 1)
y = c.step(np.random.normal(0.0, 1.0, 6))
print(y)