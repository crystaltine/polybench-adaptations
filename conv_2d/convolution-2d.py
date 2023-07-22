import torch, numpy as np

# /* Main computational kernel. The whole function will be timed,
#    including the call and return. */
# static
# void kernel_conv2d(int ni,
# 		   int nj,
# 		   DATA_TYPE POLYBENCH_2D(A,NI,NJ,ni,nj),
# 		   DATA_TYPE POLYBENCH_2D(B,NI,NJ,ni,nj))
# {
#   int i, j;
#   #pragma scop
#   #pragma omp parallel for private (j)
#   for (i = 1; i < _PB_NI - 1; ++i)
#     for (j = 1; j < _PB_NJ - 1; ++j)
#       {
# 	B[i][j]
# 	  =  0.2 * A[i-1][j-1] + 0.5 * A[i-1][j] + -0.8 * A[i-1][j+1]
# 	  + -0.3 * A[ i ][j-1] + 0.6 * A[ i ][j] + -0.9 * A[ i ][j+1]
# 	  +  0.4 * A[i+1][j-1] + 0.7 * A[i+1][j] +  0.1 * A[i+1][j+1];
#       }
#   #pragma endscop
# }
#
# This can simply be replaced with pytorch's conv2d function

def kernel_conv2d(ni: int, nj: int, A: torch.Tensor) -> torch.Tensor:
    
    """
    Parameters:
        ni: number of rows in A
        nj: number of columns in A
        A: input matrix
    """
    
    # Optimized version pytorch conv2d
    return torch.nn.functional.conv2d(A, torch.tensor([
        [ 0.2,  0.5, -0.8],
        [-0.3,  0.6, -0.9],
        [ 0.4,  0.7,  0.1]
    ]).view(1, 1, 3, 3), padding=1).view(ni-2, nj-2)