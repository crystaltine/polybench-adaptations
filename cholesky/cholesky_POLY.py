import torch
from time import perf_counter
from math import sqrt

def init_array_fixed(n: int) -> torch.Tensor:
    A = torch.zeros((n, n), dtype=torch.float64)
    for i in range(n):
        for j in range(i):
            A[i, j] = (-j % n) / n + 1
            A[j, i] = A[i, j]
        A[i, i] = n*2
        
    return A

def kernel_cholesky_original(n: int, input_matrix: torch.Tensor) -> torch.Tensor:
    """
    Original C++ Code
    ```cpp
    void kernel_cholesky(int n, DATA_TYPE POLYBENCH_2D(A,N,N,n,n)) {
        int i, j, k;


        #pragma scop
        for (i = 0; i < _PB_N; i++) {
            
            // j<i
            for (j = 0; j < i; j++) {
                for (k = 0; k < j; k++) {
                    A[i][j] -= A[i][k] * A[j][k];
                }
                A[i][j] /= A[j][j];
            }
            
            // i==j case
            for (k = 0; k < i; k++) {
                A[i][i] -= A[i][k] * A[i][k];
            }
            A[i][i] = SQRT_FUN(A[i][i]);
        }
        #pragma endscop

    }
    ```
    """
    for i in range(n):
        # j < i
        for j in range(i):
            for k in range(j):
                input_matrix[i, j] -= input_matrix[i, k] * input_matrix[j, k]
            input_matrix[i, j] /= input_matrix[j, j]
        
        # i == j case
        for k in range(i):
            input_matrix[i, i] -= input_matrix[i, k] * input_matrix[i, k]
        
        input_matrix[i, i] = input_matrix[i, i].sqrt()
    input_matrix = torch.tril(input_matrix, out=input_matrix)
    return input_matrix
        
def kernel_cholesky_slicing(n: int, input_matrix: torch.Tensor) -> torch.Tensor:
    
    """
    Original ChatGPT Code
    ```py
    import torch

    def cholesky_lower(matrix):
        # Get the dimensions of the input matrix
        n = matrix.size(0)

        # Initialize the lower triangular matrix
        lower = torch.zeros_like(matrix)

        for i in range(n):
            for j in range(i + 1):
                # Calculate the sum for the formula
                s = sum(lower[i, k] * lower[j, k] for k in range(j))

                if i == j:
                    # Diagonal elements computation
                    lower[i, j] = (matrix[i, j] - s).sqrt()
                else:
                    # Non-diagonal elements computation
                    lower[i, j] = (matrix[i, j] - s) / lower[j, j]

        return lower
    ```
    """
    
    # Initialize the lower triangular matrix
    lower = torch.zeros_like(input_matrix)
    
    for i in range(n):
        for j in range(i + 1):
            # Calculate the sum for the formula
            partial = input_matrix[i, j] - torch.sum(lower[i, :j] * lower[j, :j])
            if i == j: lower[i, i] = sqrt(partial)
            else: lower[i, j] = (1 / lower[j, j]) * partial
        
    return lower
        
def kernel_cholesky_torchapi(n: int, input_matrix: torch.Tensor) -> torch.Tensor:    
    return torch.linalg.cholesky(input_matrix)
    

def kernel_cholesky_formula(n: int, input_matrix: torch.Tensor) -> torch.Tensor:
    # FORMULA:
    # L[i,j] = (A[i,j] - Σ(L[i,k] * L_{j,k}), for k=1 to j-1) / L[j,j] (for i >= j)
    
    # Initialize the lower triangular matrix
    lower = torch.zeros_like(input_matrix)

    for i in range(n):
        for j in range(i + 1):
            # Calculate the sum for the formula
            
            # L_ii = sqrt(A_ii - Σ(L_ij^2, for j=1 to i-1))
            if i == j:
                lower[i, i] = sqrt(input_matrix[i, i] - torch.sum(lower[i, :i] ** 2))
            
            # L_ij = (1/L_jj) * (A_ij - Σ(L_ik * L_jk, for k=1 to j-1))
            else:
                lower[i, j] = (1 / lower[j, j]) * (input_matrix[i, j] - torch.sum(lower[i, :j] * lower[j, :j]))
                    
    return lower

def kernel_cholesky_algorithm_w1(n: int, input_matrix: torch.Tensor) -> torch.Tensor:
    L = torch.zeros_like(input_matrix)
    for j in range(n):

        total_sum = torch.sum(L[j][:j] * L[j][:j])
        L[j][j] = sqrt(input_matrix[j][j] - total_sum)

        for i in range(j+1, n):            
            L[i][j] = (1.0 / L[j][j] * (input_matrix[i][j] - torch.sum(L[i][:j] * L[j][:j])))
            
    return L

# Try preprocessing/postprocessing the input_matrix?
def kernel_cholesky_banachiewicz_for(n: int, input_matrix: torch.Tensor) -> torch.Tensor:
    L = torch.zeros_like(input_matrix, dtype=torch.float64)

    for i in range(n):
        
        L[i, i] = torch.sqrt(input_matrix[i, i] - torch.dot(L[i, :i], L[i, :i]))

        # With for loop:
        for j in range(i + 1, n):
            L[j, i] = (input_matrix[j, i] - torch.matmul(L[j, :i], L[i, :i])) / L[i, i]
        
    return L

def kernel_cholesky_banachiewicz_nofor(n: int, input_matrix: torch.Tensor) -> torch.Tensor:
    L = torch.zeros_like(input_matrix, dtype=torch.float64)

    for i in range(n):
        
        L[i, i] = torch.sqrt(input_matrix[i, i] - torch.dot(L[i, :i], L[i, :i]))

        # Without for loop:
        L[i+1:, i] = (input_matrix[i+1:, i] - torch.matmul(L[i+1:, :i], L[i, :i])) / L[i, i]
        
    return L

def kernel_cholesky_banachiewicz_nofor_optim(n: int, input_matrix: torch.Tensor) -> torch.Tensor:
    L = torch.zeros_like(input_matrix, dtype=torch.float64)
    
    
    # Test no for loop at all
    #L[:, :]
    #L.diagonal() -> 1d vec of diagonal
    
    for i in range(n):
        
        L[i, i] = sqrt(input_matrix[i, i] - torch.dot(L[i, :i], L[i, :i]))

        # Without for loop:
        L[i+1:, i] = (input_matrix[i+1:, i] - torch.matmul(L[i+1:, :i], L[i, :i])) / L[i, i]
        
    return L

def kernel_cholesky_crout(n: int, input_matrix: torch.Tensor) -> torch.Tensor:
    """
    ```fortran
    do i = 1, size(A,1)
        L(i,i) = sqrt(A(i,i) - dot_product(L(1:i-1,i), L(1:i-1,i)))
        L(i,i+1:) = (A(i,i+1:) - matmul(conjg(L(1:i-1,i)), L(1:i-1,i+1:))) / L(i,i)
    end do
    ```
    """
    L = torch.zeros_like(input_matrix, dtype=torch.float64)

    for j in range(n):
        L[j, j] = torch.sqrt(input_matrix[j, j] - torch.dot(L[:j, j], L[:j, j]))
        L[j, j:] = (input_matrix[j:, j] - torch.matmul(L[:j, j], L[:j, j:])) / L[j, j]
    return L.transpose(0, 1)