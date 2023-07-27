import torch
from time import perf_counter
import torch.nn.functional as F

def _get_progress_string(progress: float, length: int = 20) -> str:
    big_block = "█"
    empty_block = " "
    
    num_blocks = int(round(progress * length))
    num_empty = length - num_blocks
    
    if num_blocks == length: # so that there is no extra half block if the progress is 100%
        return "[" + big_block * num_blocks + "]"
    
    return "[" + big_block * num_blocks + empty_block * num_empty + "]"

def init_array_2d(size: int) -> torch.Tensor:
    return (torch.arange(0, size)[:, None] * torch.arange(2, size+2)[None, :] + 2) / size

def jacobi_2d_imper_original(tsteps: int, initial: torch.Tensor) -> torch.Tensor:
    print(f"Timesteps: {_get_progress_string(0)} 0/{tsteps} 0.000s", end='\r')
    start_time = perf_counter()
    for _ in range(tsteps):
        initial[1:-1, 1:-1] = 0.2 * (initial[1:-1, 1:-1] + initial[1:-1, :-2] + initial[1:-1, 2:] + initial[:-2, 1:-1] + initial[2:, 1:-1])
        print(f"Timesteps: {_get_progress_string((_+1)/tsteps)} {_+1}/{tsteps} {round((perf_counter()-start_time), 3)}s   ", end='\r')
    print("\n")
    return initial

def jacobi_2d_imper_conv(tsteps: int, initial: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor(
        [[0.0, 0.2, 0.0],
        [0.2, 0.2, 0.2],
        [0.0, 0.2, 0.0]]).view(1, 1, 3, 3)
    
    print(f"Timesteps: {_get_progress_string(0)} 0/{tsteps} 0.000s", end='\r')
    start_time = perf_counter()
    for _ in range(tsteps):
        initial[1:-1, 1:-1] = F.conv2d(initial.unsqueeze(0).unsqueeze(0), kernel).squeeze()
        print(f"Timesteps: {_get_progress_string((_+1)/tsteps)} {_+1}/{tsteps} {round((perf_counter()-start_time), 3)}s   ", end='\r')
    print("\n")
    return initial

def jacobi_2d_imper_imp3(tsteps: int, initial: torch.Tensor) -> torch.Tensor:
    print(f"Timesteps: {_get_progress_string(0)} 0/{tsteps} 0.000s", end='\r')
    start_time = perf_counter()
    for _ in range(tsteps):
        initial[1:-1, 1:-1] = (
            initial + # Center
            torch.roll(initial, 1, 0) + # Up
            torch.roll(initial, -1, 0) + # Down
            torch.roll(initial, 1, 1) + # Left
            torch.roll(initial, -1, 1) # Right
        )[1:-1, 1:-1] / 5 # Average
        print(f"Timesteps: {_get_progress_string((_+1)/tsteps)} {_+1}/{tsteps} {round((perf_counter()-start_time), 3)}s   ", end='\r')
    print("\n")
    return initial

import torch

def jacobi_2d_imper_imp4(tsteps: int, initial: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor([
        [0.0, 0.2, 0.0],
        [0.2, 0.2, 0.2],
        [0.0, 0.2, 0.0]
    ], dtype=initial.dtype)
    
    print(f"Timesteps: {_get_progress_string(0)} 0/{tsteps} 0.000s", end='\r')
    start_time = perf_counter()
    for _ in range(tsteps):
        interior = initial[1:-1, 1:-1].flatten()
        neighbors = torch.stack([
            initial[1:-1, :-2].flatten(),
            initial[1:-1, 2:].flatten(),
            initial[:-2, 1:-1].flatten(),
            initial[2:, 1:-1].flatten()
        ])

        # The kernel should be of size 9x1 to match the neighbor tensor
        kernel_flat = kernel.flatten().view(-1, 1)

        # Calculate the result only for the interior elements
        # torch.matmul using incorrect shapes
        result = (interior + 0.2 * torch.matmul(kernel_flat, neighbors)).reshape(initial[1:-1, 1:-1].shape)
        initial[1:-1, 1:-1] = result
        print(f"Timesteps: {_get_progress_string((_+1)/tsteps)} {_+1}/{tsteps} {round((perf_counter()-start_time), 3)}s   ", end='\r')
    print("\n")

    return initial
            
def jacobi_2d_imper_imp5(tsteps: int, initial: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor([
        [0.0, 0.2, 0.0],
        [0.2, 0.2, 0.2],
        [0.0, 0.2, 0.0]
    ], dtype=initial.dtype)
    print(f"Timesteps: {_get_progress_string(0)} 0/{tsteps} 0.000s", end='\r')
    start_time = perf_counter()
    
    for _ in range(tsteps):
        unfold_neighbors = initial.unfold(0, 3, 1).unfold(1, 3, 1).reshape(-1, 3, 3)
        neighbors_sum = torch.sum(unfold_neighbors * kernel.view(1, 3, 3), dim=(1, 2))
        interior = initial[1:-1, 1:-1]
        result = 0.2 * (neighbors_sum.view(initial[1:-1, 1:-1].shape))
        initial[1:-1, 1:-1] = result
        
        print(f"Timesteps: {_get_progress_string((_+1)/tsteps)} {_+1}/{tsteps} {round((perf_counter()-start_time), 3)}s   ", end='\r')
    print("\n")
        
    return initial