import torch, math, argparse, datetime
import numpy as np
import time

def _get_progress_string(progress: float, length: int = 20) -> str:
    big_block = "█"
    empty_block = "░"
    half_block = "▒"
    
    num_blocks = int(round(progress * length))
    num_empty = length - num_blocks
    
    if num_blocks == length: # so that there is no extra half block if the progress is 100%
        return big_block * num_blocks
    
    return big_block * num_blocks + empty_block * num_empty

from unused.progress_bar import _get_progress_string

# Include polybench common header

dtypes = {
    'float64': np.float64,
    'float32': np.float32,
    'double': np.double,
    'int8': np.int8,
    'int16': np.int16,
    'int32': np.int32,
    'int64': np.int64,
}

PRESETS = {
    'mini': (2, 32, 32),
    'small': (10, 500, 500),
    'standard': (50, 1000, 1000),
    'large': (50, 2000, 2000),
    'extralarge': (100, 4000, 4000)
}

# Input data type
try:
    DATA_TYPE = dtypes[input('Data type: ')]
except KeyError:
    print('Invalid data type: Supported types: {}. Proceeding with type numpy.float64'.format(list(dtypes.keys())))
    DATA_TYPE = dtypes['float64']


# Input size of the problem
use_presets = ''

while use_presets != 'y' and use_presets != 'n':    
    use_presets = input('Use preset dataset sizes? (y/n): ')

if use_presets == 'y':
    dataset_size = input('Choose a preset dataset size (mini, small, standard, large, extralarge): ')
    TMAX, NX, NY = PRESETS[dataset_size]

if use_presets == 'n':
    TMAX = int(input('TMAX (Timesteps): '))
    NX = int(input('NX (x-dim of mesh): '))
    NY = int(input('NY (y-dim of mesh): '))


# Array Initialization
def init_array(tmax: int, nx: int, ny: int, dtype: type) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sets initial values for the arrays used in a numerical simulation or computation. 
    The specific formulas used to calculate the values may vary depending on the intended application of the code.
    """
    ex = np.zeros((nx, ny), dtype=dtype)
    ey = np.zeros((nx, ny), dtype=dtype)
    hz = np.zeros((nx, ny), dtype=dtype)
    _fict_ = np.zeros(tmax, dtype=dtype)

    for i in range(tmax):
        _fict_[i] = i
    for i in range(nx):
        for j in range(ny):
            ex[i][j] = (i * (j + 1)) / nx
            ey[i][j] = (i * (j + 2)) / ny
            hz[i][j] = (i * (j + 3)) / nx
    
    return ex, ey, hz, _fict_

def kernel_fdtd_2d(tmax: int, nx: int, ny: int, ex: np.ndarray, ey: np.ndarray, hz: np.ndarray, _fict_: np.ndarray):
    """
    Apply the Finite Difference Time Domain (FDTD) algorithm to update the values of electric field components (ex and ey)
    and magnetic field component (hz) for a 2D grid over a specified number of time steps.

    Args:
        tmax (int): The maximum number of time steps to perform.
        nx (int): The number of elements in the x-direction of the grid.
        ny (int): The number of elements in the y-direction of the grid.
        ex (np.ndarray): A 2D NumPy array representing the electric field component Ex.
        ey (np.ndarray): A 2D NumPy array representing the electric field component Ey.
        hz (np.ndarray): A 2D NumPy array representing the magnetic field component Hz.
        _fict_ (np.ndarray): A 1D NumPy array representing the fictional source.

    Returns:
        None

    The function updates the values of ex, ey, and hz according to the FDTD algorithm for each time step.
    The algorithm consists of four nested loops that iterate over the grid elements and apply the update equations.
    The updates are based on the neighboring field values and the fictional source values.

    During each time step, the following steps are performed:
    1. The first loop updates the value of ey[0][j] using the fictional source value for the current time step.
    2. The second loop updates the values of ey[i][j] using the difference between hz[i][j] and hz[i-1][j].
    3. The third loop updates the values of ex[i][j] using the difference between hz[i][j] and hz[i][j-1].
    4. The fourth loop updates the values of hz[i][j] using the differences between adjacent ex and ey components.

    The function also prints the progress of the kernel execution after each time step.

    Note: The arrays ex, ey, and hz should be initialized appropriately before calling this function.
    """
    
    print(f"Timesteps: {_get_progress_string(0)} 0/{tmax}", end='\r')
    
    for t in range(tmax):
        for j in range(ny):
            ey[0][j] = _fict_[t]

        for i in range(1, nx):
            for j in range(ny):
                ey[i][j] = ey[i][j] - 0.5 * (hz[i][j] - hz[i-1][j])

        for i in range(nx):
            for j in range(1, ny):
                ex[i][j] = ex[i][j] - 0.5 * (hz[i][j] - hz[i][j-1])

        for i in range(nx - 1):
            for j in range(ny - 1):
                hz[i][j] = hz[i][j] - 0.7 * (ex[i][j+1] - ex[i][j] + ey[i+1][j] - ey[i][j])
        print(f"Timesteps: {_get_progress_string((t+1)/tmax)} {t+1}/{tmax}", end='\r')
    print("\n")
# Retrieve problem size. User input should determine these values.
tmax = TMAX
nx = NX;
ny = NY;

# Variable declaration/allocation.
# In python, array declarations are not needed. Just pass the array as an argument

# Keep in mind, in the original code, NX = nx and NY = ny

# POLYBENCH_2D_ARRAY_DECL(ex,DATA_TYPE,NX,NY,nx,ny); for ex, ey, hz
ex = ey = hz = np.zeros((nx, ny), dtype=DATA_TYPE)

#POLYBENCH_1D_ARRAY_DECL(_fict_,DATA_TYPE,TMAX,tmax);
_fict_ = np.zeros(tmax, dtype=DATA_TYPE)

# Initialize array(s).
# POLYBENCH_ARRAY returns the pointer in C, but in python, we just return the array
ex, ey, hz, _fict_ = init_array(tmax, nx, ny, DATA_TYPE)

from unused.run_function import run_function
run_function(kernel_fdtd_2d, tmax, nx, ny, ex, ey, hz, _fict_)

# Prevent dead-code elimination. All live-out data must be printed
#    by the function call in argument.
#
# polybench_prevent_dce(
#    print_array(nx, ny, POLYBENCH_ARRAY(ex),
#    POLYBENCH_ARRAY(ey),
#    POLYBENCH_ARRAY(hz)
# ));

# Garbage collection, not needed in python
# POLYBENCH_FREE_ARRAY(ex);
# POLYBENCH_FREE_ARRAY(ey);
# POLYBENCH_FREE_ARRAY(hz);
# POLYBENCH_FREE_ARRAY(_fict_);
