import torch
from time import perf_counter

def _get_progress_string(progress: float, length: int = 20) -> str:
    big_block = "█"
    empty_block = "\x1B[2m█\033[0m"
    
    num_blocks = int(round(progress * length))
    num_empty = length - num_blocks
    
    if num_blocks == length: # so that there is no extra half block if the progress is 100%
        return "[" + big_block * num_blocks + "]"
    
    return "[" + big_block * num_blocks + empty_block * num_empty + "]"

def init_array(n: int) -> torch.Tensor:
    A = torch.empty((n, n), dtype=torch.float64)
    for i in range(n):
        for j in range(i+1):
            A[i][j] = (-j % n) / n + 1
        for j in range(i+1, n):
            A[i][j] = 0
        A[i][i] = 1 # Diagonals are 1
    return A

def kernel_lu_original(n: int, input_matrix: torch.Tensor) -> None:
    """
    Original C Code:
    ```c
    void kernel_lu(int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n)) {
        int i, j, k;

        #pragma scop
        for (i = 0; i < _PB_N; i++) {
            for (j = 0; j <i; j++) {
                for (k = 0; k < j; k++) {
                    A[i][j] -= A[i][k] * A[k][j];
                }
                A[i][j] /= A[j][j];
            }
            for (j = i; j < _PB_N; j++) {
                for (k = 0; k < i; k++) {
                    A[i][j] -= A[i][k] * A[k][j];
                }
            }
        }
        #pragma endscop
    }
    ```
    """
    print(f"Operations: {_get_progress_string(0)} 0/0 0.000s", end='\r')
    total_time = 0
    for i in range(n):
        start_time = perf_counter()
        for j in range(i):
            for k in range(j):
                input_matrix[i][j] -= input_matrix[i][k] * input_matrix[k][j]
            input_matrix[i][j] /= input_matrix[j][j]
        for j in range(i, n):
            for k in range(i):
                input_matrix[i][j] -= input_matrix[i][k] * input_matrix[k][j]
                
        total_time += perf_counter() - start_time
        print(f"Operations: {_get_progress_string((i+1)/n)} {i+1}/{n} {round((total_time*1000), 3)}ms   ", end='\r')
    print("\n")
                
def kernel_lu_slicing(n: int, input_matrix: torch.Tensor) -> None:
    print(f"Operations: {_get_progress_string(0)} 0/0 0.000s", end='\r')
    total_time = 0
    for i in range(n):
        start_time = perf_counter()
        
        for j in range(i):
            input_matrix[i, j] -= torch.sum(input_matrix[i, :j] * input_matrix[:j, j])
        # diagonals are 1, so we don't need to divide by A[j][j]
        
        for j in range(i, n):
            input_matrix[i, j] -= torch.sum(input_matrix[i, :i] * input_matrix[:i, j])
        
        total_time += perf_counter() - start_time
        print(f"Operations: {_get_progress_string((i+1)/n)} {i+1}/{n} {round((total_time*1000), 3)}ms   ", end='\r')
    print("\n")
        
def kernel_lu_tril(n: int, input_matrix: torch.Tensor) -> None:
    print(f"Operations: {_get_progress_string(0)} 0/0 0.000s", end='\r')
    total_time = 0
    lu_matrix = input_matrix.clone()

    for i in range(n):
        
        start_time = perf_counter()
        
        for j in range(i):
            lu_matrix[i, j] -= torch.sum(lu_matrix[i, :j] * lu_matrix[:j, j])
            lu_matrix[i, j] /= lu_matrix[j, j]

        for j in range(i, n):
            lu_matrix[i, j] -= torch.sum(lu_matrix[i, :i] * lu_matrix[:i, j])
    
        total_time += perf_counter() - start_time
        print(f"Operations: {_get_progress_string((i+1)/(n+1))} {i+1}/{(n+1)} {round((total_time*1000), 3)}ms   ", end='\r')
        
    input_matrix = torch.tril(lu_matrix, diagonal=-1) + torch.eye(n, dtype=lu_matrix.dtype, device=lu_matrix.device)
    total_time += perf_counter() - start_time
    print(f"Operations: {_get_progress_string(1)} {n+1}/{(n+1)} {round((total_time*1000), 3)}ms   ", end='\r')
    print("\n")


        
def kernel_lu_arange_calc(n: int, input_matrix: torch.Tensor) -> None:
    print(f"Operations: {_get_progress_string(0)} 0/0 0.000s", end='\r')
    total_time = 0
    for i in range(2, n):
        
        start_time = perf_counter()

        # Solve lower triangle values
        input_matrix[i, 1:i] = torch.arange(2 - 1/n, 2-(1/n)*(i), -1/n)[:i-1]
        
        total_time += perf_counter() - start_time
        print(f"Operations: {_get_progress_string((i+1)/(n+1))} {i+1}/{n+1} {round((total_time*1000), 3)}ms   ", end='\r')
    
    # boundaries (diagonal and edge)
    input_matrix[:, 0] = 1
    input_matrix.diagonal().fill_(1)
    print(f"Operations: {_get_progress_string(1)} {n+1}/{(n+1)} {round((total_time*1000), 3)}ms   ", end='\r')
    print("\n")
    
def kernel_lu_torchapi(n: int, input_matrix: torch.Tensor) -> None:
    print(f"Operations: {_get_progress_string(0)} 0/0 0.000s", end='\r')
    total_time = 0
    start_time = perf_counter()
    
    # NOTE: this only computes the lower triangular matrix
    input_matrix = torch.linalg.lu_factor(input_matrix)[1]
    
    total_time += perf_counter() - start_time
    print(f"Operations: {_get_progress_string(1)} 1/1 {round((total_time*1000), 3)}ms   ", end='\r')
    print("\n")
