# Original C code
# polybenchgpu/OpenMP/stencils/convolution-3d/convolution-3d.c
# https://github.com/sgrauerg/polybenchGpu/blob/master/OpenMP/stencils/convolution-3d/convolution-3d.c
#
# /*
# Main computational kernel. The whole function will be timed,
# including the call and return. */
# static
# void kernel_conv2d(int ni,
# 		   int nj,
# 		   int nk,
# 		   DATA_TYPE POLYBENCH_3D(A,NI,NJ,NK,ni,nj,nk),
# 		   DATA_TYPE POLYBENCH_3D(B,NI,NJ,NK,ni,nj,nk))
# {
#   int i, j, k;
#   #pragma scop
#   #pragma omp parallel
#   {
#     #pragma omp for private (j,k) collapse(2)
#     for (i = 1; i < _PB_NI - 1; ++i)
#       for (j = 1; j < _PB_NJ - 1; ++j)
# 	for (k = 1; k < _PB_NK - 1; ++k)
# 	  {
#              B[i][j][k]

# ??? What is the kernel used here? XXX

# Idea: some of the indicies below are wrong, it should be:
#   If centered on the center of a 3x3x3 cube, then cells to look at are
#   The diagonals (X) of the top layer, and
#   The diagonals (X) of the bottom layer, and
#   The diagonals (X) of the middle layer, and

# So it should be [j-1],[j],[j+1] for all of [i+-1][k+-1] and [i][k] (5 on each layer)

# 	       =  2 * A[i-1][j-1][k-1]  +  4 * A[i+1][j-1][k-1]
# 	       +  5 * A[i-1][j-1][k-1]  +  7 * A[i+1][j-1][k-1]
# 	       + -8 * A[i-1][j-1][k-1]  + 10 * A[i+1][j-1][k-1]
# 	       + -3 * A[ i ][j-1][ k ]
# 	       +  6 * A[ i ][ j ][ k ]
# 	       + -9 * A[ i ][j+1][ k ]
# 	       +  2 * A[i-1][j-1][k+1]  +  4 * A[i+1][j-1][k+1]
# 	       +  5 * A[i-1][ j ][k+1]  +  7 * A[i+1][ j ][k+1]
# 	       + -8 * A[i-1][j+1][k+1]  + 10 * A[i+1][j+1][k+1];
#            }
#   }
#   #pragma endscop
# }
#
# simply replace with pytorch's conv3d function

import torch
import torch.nn.functional as F
from time import perf_counter

def _get_progress_string(progress: float, length: int = 20) -> str:
    big_block = "█"
    empty_block = "\x1B[30m█\033[0m"
    
    num_blocks = int(round(progress * length))
    num_empty = length - num_blocks
    
    if num_blocks == length: # so that there is no extra half block if the progress is 100%
        return "[" + big_block * num_blocks + "]"
    
    return "[" + big_block * num_blocks + empty_block * num_empty + "]"

# Original C++ code for reference
# /* Array initialization. */
# static
# void init_array (int ni, int nj, int nk,
# 		 DATA_TYPE POLYBENCH_3D(A,NI,NJ,NK,ni,nj,nk))
# {
#   int i, j, k;
# 
#   for (i = 0; i < ni; i++)
#     for (j = 0; j < nj; j++)
#       for (k = 0; j < nk; k++)
# 	{
# 	  A[i][j][k] = i % 12 + 2 * (j % 7) + 3 * (k % 13);
# 	}
# }
RESET = '\033[0m'
COLOR_CODE_DIM = "\x1B[2m" # 30: Gray
def init_array(ni: int, nj: int, nk: int) -> torch.Tensor:
    a = torch.zeros((ni, nj, nk), dtype=torch.float64)
    print(COLOR_CODE_DIM + "Initalizing values...\n" + RESET)
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                a[i][j][k] = i % 12 + 2 * (j % 7) + 3 * (k % 13)
    
    return a

def kernel_conv3d_direct_translation(ni: int, nj: int, nk: int, A: torch.Tensor) -> torch.Tensor:
    
    max_ops = (ni - 2) * (nj - 2) * (nk - 2)
    ops = 0
    print(f"Operations: {_get_progress_string(0)} 0/{max_ops} 0.000s", end='\r')
    start_time = perf_counter()
    
    temp = torch.zeros((ni, nj, nk), dtype=torch.float64)
    
    for i in range(1, ni - 1):
        for j in range(1, nj - 1):
            for k in range(1, nk - 1):
                temp[i][j][k] = \
                    2 * A[i-1][j-1][k-1]  +  4 * A[i+1][j-1][k-1] + \
                    5 * A[i-1][j-1][k-1]  +  7 * A[i+1][j-1][k-1] + \
                   -8 * A[i-1][j-1][k-1]  + 10 * A[i+1][j-1][k-1] + \
                   -3 * A[ i ][j-1][ k ] + \
                    6 * A[ i ][ j ][ k ] + \
                   -9 * A[ i ][j+1][ k ] + \
                    2 * A[i-1][j-1][k+1]  +  4 * A[i+1][j-1][k+1] + \
                    5 * A[i-1][ j ][k+1]  +  7 * A[i+1][ j ][k+1] + \
                   -8 * A[i-1][j+1][k+1]  + 10 * A[i+1][j+1][k+1]
                   
                ops += 1
                print(f"Operations: {_get_progress_string((ops)/max_ops)} {ops}/{max_ops} {round((perf_counter()-start_time), 3)}s   ", end='\r')
    print("\n")
    A[1:-1, 1:-1, 1:-1] = temp[1:-1, 1:-1, 1:-1]
    return A

def kernel_conv3d_torchconv(ni: int, nj: int, nk: int, A: torch.Tensor) -> torch.Tensor:
    # Create a 3x3x3 kernel to represent the coefficients for each neighbor
    kernel = torch.tensor([
        [[-1, 0, 2], 
         [0, 0, 5], 
         [0, 0, -8]], 
        [[0, -3, 0], 
         [0, 6, 0], 
         [0, -9, 0]], 
        [[21, 0, 4], 
         [0, 0, 7], 
         [0, 0, 10]]
    ], dtype=torch.float64)
    
    print(f"Operations: {_get_progress_string(0)} 0/1 0.000s", end='\r')
    start_time = perf_counter()
    
    # Convolve the kernel with the input tensor
    A[1:-1, 1:-1, 1:-1] = F.conv3d(
        input=A.unsqueeze(0).unsqueeze(0), # add batch and channel dimensions
        weight=kernel.unsqueeze(0).unsqueeze(0), # add batch and channel dimensions
    ).squeeze(0).squeeze(0) # remove batch and channel dimensions
    
    print(f"Operations: {_get_progress_string(1)} 1/1 {round((perf_counter()-start_time), 3)}s   ", end='\r')
    print("\n")
    
    return A

def kernel_conv3d_roll(ni: int, nj: int, nk: int, A: torch.Tensor) -> torch.Tensor:

    # Roll the tensor a lot of times and sum
    # each of them up to match the kernel
    
    kernel = torch.tensor([
        [[-1, 0, 2], 
         [0, 0, 5], 
         [0, 0, -8]], 
        [[0, -3, 0], 
         [0, 6, 0], 
         [0, -9, 0]], 
        [[21, 0, 4], 
         [0, 0, 7], 
         [0, 0, 10]]
    ], dtype=torch.float64)
    
    A[1:-1, 1:-1, 1:-1] = (
        A[:-2, :-2, :-2] * kernel[0, 0, 0] +
        A[:-2, :-2, 2:] * kernel[0, 0, 2] +
        A[:-2, 1:-1, 2:] * kernel[0, 1, 2] +
        A[:-2, 2:, 2:] * kernel[0, 2, 2] +
        
        A[1:-1, :-2, 1:-1] * kernel[1, 0, 1] +
        A[1:-1, 1:-1, 1:-1] * kernel[1, 1, 1] +
        A[1:-1, 2:, 1:-1] * kernel[1, 2, 1] +

        A[2:, :-2, :-2] * kernel[2, 0, 0] +
        A[2:, :-2, 2:] * kernel[2, 0, 2] +
        A[2:, 1:-1, 2:] * kernel[2, 1, 2] +
        A[2:, 2:, 2:] * kernel[2, 2, 2]
    )
                
    return A






