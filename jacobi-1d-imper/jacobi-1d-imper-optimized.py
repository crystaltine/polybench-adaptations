"""

Original C code

/* POLYBENCH/GPU-OPENMP
 *
 * This file is a part of the Polybench/GPU-OpenMP suite
 *
 * Contact:
 * William Killian <killian@udel.edu>
 * 
 * Copyright 2013, The University of Delaware
 */
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 100x10000. */
#include "jacobi-1d-imper.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_1D(A,N,n),
		 DATA_TYPE POLYBENCH_1D(B,N,n))
{
  int i;

  for (i = 0; i < n; i++)
      {
	A[i] = ((DATA_TYPE) i+ 2) / n;
	B[i] = ((DATA_TYPE) i+ 3) / n;
      }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(A,N,n))

{
  int i;

  for (i = 0; i < n; i++)
    {
      fprintf(stderr, DATA_PRINTF_MODIFIER, A[i]);
      if (i % 20 == 0) fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_jacobi_1d_imper(int tsteps,
			    int n,
			    DATA_TYPE POLYBENCH_1D(A,N,n),
			    DATA_TYPE POLYBENCH_1D(B,N,n))
{
  int t, i, j;

  #pragma scop
  #pragma omp parallel
  {
    for (t = 0; t < _PB_TSTEPS; t++)
      {
        #pragma omp for
        for (i = 1; i < _PB_N - 1; i++)
	  B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1]);
	#pragma omp for
	for (j = 1; j < _PB_N - 1; j++)
	    A[j] = B[j];
      }
  }
  #pragma endscop
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_1D_ARRAY_DECL(A, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(B, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_jacobi_1d_imper (tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}
"""

import numpy as np, torch

#
# /* Array initialization. */
#static
#void init_array (int n,
#		 DATA_TYPE POLYBENCH_1D(A,N,n),
#		 DATA_TYPE POLYBENCH_1D(B,N,n))
#{
#  int i;
#
#  for (i = 0; i < n; i++)
#      {
#	A[i] = ((DATA_TYPE) i+ 2) / n;
#	B[i] = ((DATA_TYPE) i+ 3) / n;
#      }
#}

def init_arrays(size: int, optimize=True) -> tuple:
    
    """
    Returns default initialized arrays A and B
    """
    if optimize:
        # Using numpy/PyTorch to initialize arrays
        return np.arange(2, size+2) / size, np.arange(3, size+3) / size
    
# /*
#    including the call and return. */
# static
# void kernel_jacobi_1d_imper(int tsteps,
# 			    int n,
# 			    DATA_TYPE POLYBENCH_1D(A,N,n),
# 			    DATA_TYPE POLYBENCH_1D(B,N,n))
# {
#   int t, i, j;

#   #pragma scop
#   #pragma omp parallel
#   {
#     for (t = 0; t < _PB_TSTEPS; t++)
#     {
#         #pragma omp for
#         for (i = 1; i < _PB_N - 1; i++)
# 	          B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1]);
#
# 	      #pragma omp for
# 	      for (j = 1; j < _PB_N - 1; j++)
# 	          A[j] = B[j];
#     }
#   }
#   #pragma endscop
# }
from unused.progress_bar import _get_progress_string
def kernel_jacobi_1d_imper(tsteps: int, initial: torch.Tensor | np.ndarray) -> None:
    
    """
    Main computational kernel. The whole function will be timed,
    including the call and return.
    
    The function is basically just a sliding window calculating a moving average of width 3.
    """
    
    print(f"Timesteps: {_get_progress_string(0)} 0/{tsteps}", end='\r')
    for _ in range(tsteps):
        initial[1:-1] = 0.33333 * (initial[:-2] + initial[1:-1] + initial[2:])
        print(f"Timesteps: {_get_progress_string((_+1)/tsteps)} {_+1}/{tsteps}", end='\r')
    print('\n')
              
        
from unused.run_function import run_function

SIZE = int(input("Enter size: "))
TSTEPS = int(input("Enter timesteps: "))

A, B = init_arrays(SIZE)

# TODO: why do we have 2 arrays? The C code looks like it only uses one, and just overwrites the other.

run_function(kernel_jacobi_1d_imper, TSTEPS, A)