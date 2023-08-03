import torch
from time import perf_counter

def init_array(n: int) -> torch.Tensor:
    A = torch.empty((n, n), dtype=torch.float64)
    for i in range(n):
        for j in range(i+1):
            A[i][j] = (-j % n) / n + 1
        for j in range(i+1, n):
            A[i][j] = 0
        A[i][i] = n*2 # Diagonals are 1
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
    
    for i in range(n):
        
        for j in range(i):
            for k in range(j):
                input_matrix[i][j] -= input_matrix[i][k] * input_matrix[k][j]
            input_matrix[i][j] /= input_matrix[j][j]
        for j in range(i, n):
            for k in range(i):
                input_matrix[i][j] -= input_matrix[i][k] * input_matrix[k][j]
                                
def kernel_lu_slicing(n: int, input_matrix: torch.Tensor) -> None:
    for i in range(n):
        start_time = perf_counter()
        
        for j in range(i):
            input_matrix[i, j] -= torch.sum(input_matrix[i, :j] * input_matrix[:j, j]) / input_matrix[i, i]
        
        for j in range(i, n):
            input_matrix[i, j] -= torch.sum(input_matrix[i, :i] * input_matrix[:i, j]) / input_matrix[i, i]
        
        
def kernel_lu_tril(n: int, input_matrix: torch.Tensor) -> None:
    lu_matrix = input_matrix.clone()

    for i in range(n):

        for j in range(i):
            lu_matrix[i, j] -= torch.sum(lu_matrix[i, :j] * lu_matrix[:j, j])
            lu_matrix[i, j] /= lu_matrix[j, j]

        #for j in range(i, n):
        #    lu_matrix[i, j] -= torch.sum(lu_matrix[i, :i] * lu_matrix[:i, j])
        lu_matrix[i, :i] -= torch.sum(lu_matrix[i, :i] * lu_matrix[:i, :i])
            
    input_matrix = torch.tril(lu_matrix, diagonal=-1) + torch.eye(n, dtype=lu_matrix.dtype, device=lu_matrix.device)

        
def kernel_lu_arange_calc_TEST(n: int, input_matrix: torch.Tensor) -> None:
    
    for i in range(2, n):
        # Solve lower triangle values
        input_matrix[i, 1:i] = 2*n* torch.arange(2 - 1/n, 2-(1/n)*(i), -1/n)[:i-1] / input_matrix[i, i]
        
    # boundaries (diagonal and edge)
    input_matrix[:, 0] = 1
    input_matrix.diagonal().fill_(2*n)
    
def kernel_lu_torchapi(n: int, input_matrix: torch.Tensor) -> None:
    # NOTE: this only computes the lower triangular matrix
    input_matrix = torch.linalg.lu_factor(input_matrix)[1] / input_matrix[0, 0]
    
def kernel_lu_nofor(n: int, input_matrix: torch.Tensor) -> None:
    for i in range(n):
        
        for j in range(i):
            input_matrix[i, j] -= torch.sum(input_matrix[i, :j] * input_matrix[:j, j])
        #input_matrix[i, :i] -= torch.sum(input_matrix[i, [slice(j) for j in range(i)]] * input_matrix[[slice(j) for j in range(i)], :i], dim=0)        
        
        # for j in range(i, n):
        #     input_matrix[i, j] -= torch.sum(input_matrix[i, :i] * input_matrix[:i, j])
        input_matrix[i, i:] -= torch.sum(input_matrix[i, :i].unsqueeze(1) * input_matrix[:i, i:], dim=0)
        # input_matrix[i, :i] /= input_matrix[i, i]
            
def lu_decomposition_partial_pivoting(n: int, input_matrix: torch.Tensor) -> torch.Tensor:
    L = torch.zeros((n, n), dtype=input_matrix.dtype)

    for k in range(n - 1):
        # Preprocess by partial pivoting the matrix
        pivot_row = torch.argmax(torch.abs(input_matrix[k:, k])) + k
        L[[k, pivot_row]] = L[[pivot_row, k]]
        L[k+1:n, k:] -= L[k, k:]
                
    return L
