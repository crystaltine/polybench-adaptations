import numpy as np, torch
from unused.progress_bar import _get_progress_string

# /* Array initialization. */
# static
# void init_array (int n,
# 		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
# 		 DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
# {
#   int i, j;

#   for (i = 0; i < n; i++)
#     for (j = 0; j < n; j++)
#       {
# 	A[i][j] = ((DATA_TYPE) i*(j+2) + 2) / n;
# 	B[i][j] = ((DATA_TYPE) i*(j+3) + 3) / n;
#       }
# }

def init_array_2d(size: int) -> tuple:
    return (np.arange(0, size)[:, None] * np.arange(2, size+2)[None, :] + 2) / size, \
        (np.arange(0, size)[:, None] * np.arange(3, size+3)[None, :] + 3) / size
        
def jacobi_2d_imper(tsteps: int, initial: np.ndarray | torch.Tensor) -> None:
    """
    ## Original C:
    
    ```c
    /* Main computational kernel. The whole function will be timed,
    including the call and return. */
    static void kernel_jacobi_2d(
        int tsteps,
        int n,
        DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
        DATA_TYPE POLYBENCH_2D(B,N,N,n,n)
        ) {
        int t, i, j;

        #pragma scop
        for (t = 0; t < _PB_TSTEPS; t++)
            {
            for (i = 1; i < _PB_N - 1; i++)
                for (j = 1; j < _PB_N - 1; j++)
                    B[i][j] = SCALAR_VAL(0.2) * (A[i][j] + A[i][j-1] + A[i][1+j] + A[1+i][j] + A[i-1][j]);
            for (i = 1; i < _PB_N - 1; i++)
                for (j = 1; j < _PB_N - 1; j++)
                    A[i][j] = SCALAR_VAL(0.2) * (B[i][j] + B[i][j-1] + B[i][1+j] + B[1+i][j] + B[i-1][j]);
            }
        #pragma endscop
    }
    
    Assumtions: 
    + DATA_TYPE POLYBENCH_2D(A|B,N,N,n,n) will be represented by an ndarray of shape (n, n)
    + _PB_N is probably n (the size of the array)
    
    Manipulates the given initial array in-place.
    """
    
    print(f"Timesteps: {_get_progress_string(0)} 0/{tsteps}", end='\r')
    
    for _ in range(tsteps):
        initial[1:-1, 1:-1] = 0.2 * (initial[1:-1, 1:-1] + initial[1:-1, :-2] + initial[1:-1, 2:] + initial[:-2, 1:-1] + initial[2:, 1:-1])
    
        print(f"Timesteps: {_get_progress_string((_+1)/tsteps)} {_+1}/{tsteps}", end='\r')
    print('\n')
    
from unused.run_function import run_function

SIZE = int(input("Enter size: "))
TSTEPS = int(input("Enter timesteps: "))

A, B = init_array_2d(SIZE)

# TODO: why do we have 2 arrays? The C code looks like it only uses one, and just overwrites the other.

run_function(jacobi_2d_imper, TSTEPS, A)
    
    