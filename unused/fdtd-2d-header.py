TMAX = 0
NX = 0
NY = 0

dataset_size = input("Please enter the dataset size: ")

# Define the possible dataset sizes
if "MINI_DATASET":
    TMAX = 2
    NX = 32
    NY = 32

elif "SMALL_DATASET":
    TMAX = 10
    NX = 500
    NY = 500

elif "LARGE_DATASET":
    TMAX = 50
    NX = 2000
    NY = 2000

elif "EXTRALARGE_DATASET":
    TMAX = 100
    NX = 4000
    NY = 4000

else: # assume standard dataset
    TMAX = 50
    NX = 1000
    NY = 1000

# Print the selected dataset sizes
print("TMAX:", TMAX)
print("NX:", NX)
print("NY:", NY)

define POLYBENCH_1D_ARRAY_DECL(var, type, dim1, ddim1)		\
  type POLYBENCH_1D(POLYBENCH_DECL_VAR(var), dim1, ddim1); \
  var = POLYBENCH_ALLOC_1D_ARRAY(POLYBENCH_C99_SELECT(dim1, ddim1), type);

#  define POLYBENCH_2D_ARRAY_DECL(var, type, dim1, dim2, ddim1, ddim2)	\
  type POLYBENCH_2D(POLYBENCH_DECL_VAR(var), dim1, dim2, ddim1, ddim2); \
  var = POLYBENCH_ALLOC_2D_ARRAY(POLYBENCH_C99_SELECT(dim1, ddim1), POLYBENCH_C99_SELECT(dim2, ddim2), type);

#  define POLYBENCH_3D_ARRAY_DECL(var, type, dim1, dim2, dim3, ddim1, ddim2, ddim3) \
  type POLYBENCH_3D(POLYBENCH_DECL_VAR(var), dim1, dim2, dim3, ddim1, ddim2, ddim3); \
  var = POLYBENCH_ALLOC_3D_ARRAY(POLYBENCH_C99_SELECT(dim1, ddim1), POLYBENCH_C99_SELECT(dim2, ddim2), POLYBENCH_C99_SELECT(dim3, ddim3), type);

#  define POLYBENCH_4D_ARRAY_DECL(var, type, dim1, dim2, dim3, dim4, ddim1, ddim2, ddim3, ddim4) \
  type POLYBENCH_4D(POLYBENCH_DECL_VAR(var), dim1, dim2, ,dim3, dim4, ddim1, ddim2, ddim3, ddim4); \
  var = POLYBENCH_ALLOC_4D_ARRAY(POLYBENCH_C99_SELECT(dim1, ddim1), POLYBENCH_C99_SELECT(dim2, ddim2), POLYBENCH_C99_SELECT(dim3, ddim3), POLYBENCH_C99_SELECT(dim4, ddim4), type);

#  define POLYBENCH_5D_ARRAY_DECL(var, type, dim1, dim2, dim3, dim4, dim5, ddim1, ddim2, ddim3, ddim4, ddim5) \
  type POLYBENCH_5D(POLYBENCH_DECL_VAR(var), dim1, dim2, dim3, dim4, dim5, ddim1, ddim2, ddim3, ddim4, ddim5); \
  var = POLYBENCH_ALLOC_5D_ARRAY(POLYBENCH_C99_SELECT(dim1, ddim1), POLYBENCH_C99_SELECT(dim2, ddim2), POLYBENCH_C99_SELECT(dim3, ddim3), POLYBENCH_C99_SELECT(dim4, ddim4), POLYBENCH_C99_SELECT(dim5, ddim5), type);