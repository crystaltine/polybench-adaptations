"""
Multiple implementations of `kernel_fdtd_2d` (see fdtd_2d_optimized.py) for one such.
Using: Pytorch
"""

import torch, numpy as np
from time import perf_counter

def _get_progress_string(progress: float, length: int = 20) -> str:
    big_block = "█"
    empty_block = "\x1B[30m█\033[0m"
    
    num_blocks = int(round(progress * length))
    num_empty = length - num_blocks
    
    if num_blocks == length: # so that there is no extra half block if the progress is 100%
        return big_block * num_blocks
    
    return big_block * num_blocks + empty_block * num_empty

def init_array(tmax: int, nx: int, ny: int, dtype: type) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sets initial values for the arrays used in a numerical simulation or computation. 
    The specific formulas used to calculate the values may vary depending on the intended application of the code.
    """
    ex = torch.zeros(nx, ny, dtype=dtype)
    ey = torch.zeros(nx, ny, dtype=dtype)
    hz = torch.zeros(nx, ny, dtype=dtype)
    _fict_ = torch.zeros(tmax, dtype=dtype)

    for i in range(tmax):
        _fict_[i] = i
    for i in range(nx):
        for j in range(ny):
            ex[i][j] = (i * (j + 1)) / nx
            ey[i][j] = (i * (j + 2)) / ny
            hz[i][j] = (i * (j + 3)) / nx
    
    return ex, ey, hz, _fict_

def kernel_fdtd_2d_original(tmax: int, nx: int, ny: int, ex: torch.Tensor, ey: torch.Tensor, hz: torch.Tensor, _fict_: torch.Tensor) -> tuple:
    print(f"Timesteps: {_get_progress_string(0)} 0/{tmax} 0.000s", end='\r')
    start_time = perf_counter()
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
        print(f"Timesteps: {_get_progress_string((t+1)/tmax)} {t+1}/{tmax} {round((perf_counter()-start_time), 3)}s   ", end='\r')
    print("\n")
    
    return ex, ey, hz

def kernel_fdtd_2d_some_optimization(tmax: int, nx: int, ny: int, ex: torch.Tensor, ey: torch.Tensor, hz: torch.Tensor, _fict_: torch.Tensor) -> tuple:
    """ See `fdtd_2d_optimized.kernel_fdtd_2d` for full docstring """

    print(f"Timesteps: {_get_progress_string(0)} 0/{tmax} 0.000s", end='\r')
    start_time = perf_counter()
    # From fdtd_2d_optimized.py
    for t in range(tmax):
        ey[0, :] = _fict_[t]
        for i in range(1, nx):
            ey[i, :] = ey[i, :] - 0.5 * (hz[i, :] - hz[i-1, :])
        for j in range(1, ny):
            ex[:, j] = ex[:, j] - 0.5 * (hz[:, j] - hz[:, j-1])
        hz[:-1, :-1] = hz[:-1, :-1] - 0.7 * (ex[:-1, 1:] - ex[:-1, :-1] + ey[1:, :-1] - ey[:-1, :-1])
        print(f"Timesteps: {_get_progress_string((t+1)/tmax)} {t+1}/{tmax} {round((perf_counter()-start_time), 3)}s   ", end='\r')
    print("\n")
        
    return ex, ey, hz
        
def kernel_fdtd_2d_no_forloop(tmax: int, nx: int, ny: int, ex: torch.Tensor, ey: torch.Tensor, hz: torch.Tensor, _fict_: torch.Tensor) -> tuple:
    print(f"Timesteps: {_get_progress_string(0)} 0/{tmax} 0.000s", end='\r')
    start_time = perf_counter()
    # Eliminated for loops
    for t in range(tmax):
        ey[0, :] = _fict_[t]
        ey[1:nx, :] -= 0.5 * (hz[1:nx, :] - hz[:nx-1, :])
        ex[:, 1:ny] -= 0.5 * (hz[:, 1:ny] - hz[:, :ny-1])
        hz[:-1, :-1] -= 0.7 * (ex[:-1, 1:] - ex[:-1, :-1] + ey[1:, :-1] - ey[:-1, :-1])
        print(f"Timesteps: {_get_progress_string((t+1)/tmax)} {t+1}/{tmax} {round((perf_counter()-start_time), 3)}s   ", end='\r')
    print("\n")
        
    return ex, ey, hz

# @fixed
def kernel_fdtd_2d_diff(tmax: int, nx: int, ny: int, ex: torch.Tensor, ey: torch.Tensor, hz: torch.Tensor, _fict_: torch.Tensor) -> tuple:
    print(f"Timesteps: {_get_progress_string(0)} 0/{tmax} 0.000s", end='\r')
    start_time = perf_counter()
    # Additional implementation using torch.diff
    for t in range(tmax):
        ey[0, :] = _fict_[t]
        
        # Update ey for the x-direction
        ey[1:nx, :] -= 0.5 * torch.diff(hz, dim=0)[:nx-1, :]
        
        # Update ex for the y-direction
        ex[:, 1:ny] -= 0.5 * torch.diff(hz, dim=1)[:, :ny-1]
        
        # Update hz
        hz[:-1, :-1] -= 0.7 * (torch.diff(ex, dim=1)[:-1, :] + torch.diff(ey, dim=0)[:, :-1])
        
        print(f"Timesteps: {_get_progress_string((t+1)/tmax)} {t+1}/{tmax} {round((perf_counter()-start_time), 3)}s   ", end='\r')
    print("\n")
        
    return ex, ey, hz
        
def kernel_fdtd_2d_roll(tmax: int, nx: int, ny: int, ex: torch.Tensor, ey: torch.Tensor, hz: torch.Tensor, _fict_: torch.Tensor) -> tuple:
    print(f"Timesteps: {_get_progress_string(0)} 0/{tmax} 0.000s", end='\r')
    for t in range(tmax):
        start_time = perf_counter()
        # Assign _fict_[t] to the first row of ey
        ey[0, :] = _fict_[t]

        # Update ey for the x-direction using torch.roll
        ey[1:nx, :] = ey[1:nx, :] - 0.5 * (hz[1:nx, :] - torch.roll(hz, shifts=1, dims=0)[1:nx, :])

        # Update ex for the y-direction using torch.roll
        ex[:, 1:ny] = ex[:, 1:ny] - 0.5 * (hz[:, 1:ny] - torch.roll(hz, shifts=1, dims=1)[:, 1:ny])

        # Update hz using torch.roll
        hz[:-1, :-1] = hz[:-1, :-1] - 0.7 * (ex[:-1, 1:] - ex[:-1, :-1] + ey[1:, :-1] - ey[:-1, :-1])
        
        total_time += perf_counter() - start_time
        print(f"Timesteps: {_get_progress_string((t+1)/tmax)} {t+1}/{tmax} {round((total_time), 3)}s   ", end='\r')
    print("\n")

    return ex, ey, hz
        
def kernel_fdtd_2d_index_shift(tmax: int, nx: int, ny: int, ex: torch.Tensor, ey: torch.Tensor, hz: torch.Tensor, _fict_: torch.Tensor) -> tuple:
    print(f"Timesteps: {_get_progress_string(0)} 0/{tmax} 0.000s", end='\r')
    total_time = 0
    # Additional implementation using torch.flip
    for t in range(tmax):
        start_time = perf_counter()
        # Assign _fict_[t] to the first row of ey
        ey[0, :] = _fict_[t]

        # Update ey for the x-direction
        ey[1:nx, :] -= 0.5 * (hz[1:nx, :] - hz[:nx-1, :])

        # Update ex for the y-direction
        ex[:, 1:ny] -= 0.5 * (hz[:, 1:ny] - hz[:, :ny-1])

        # Create slices for shifted versions of ex and ey for the hz update
        ex_shifted = ex[:-1, 1:]
        ey_shifted = ey[1:, :-1]

        # Update hz using tensor operations
        hz[:-1, :-1] -= 0.7 * (ex_shifted - ex[:-1, :-1] + ey_shifted - ey[:-1, :-1])

        total_time += perf_counter() - start_time
        print(f"Timesteps: {_get_progress_string((t+1)/tmax)} {t+1}/{tmax} {round((total_time), 3)}s   ", end='\r')
    print("\n")
        
    return ex, ey, hz