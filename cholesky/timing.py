from time import perf_counter

from cholesky_POLY import (
    init_array_fixed,
    kernel_cholesky_formula,
    kernel_cholesky_original,
    kernel_cholesky_slicing,
    kernel_cholesky_torchapi,
    kernel_cholesky_algorithm_w1,
    kernel_cholesky_banachiewicz_for,
    kernel_cholesky_banachiewicz_nofor,
    kernel_cholesky_crout,
    kernel_cholesky_banachiewicz_nofor_optim
)

# Run each function on larger and larger inputs
test_array_sizes = [5,   10,  15, 20, 25, 30, 50, 100]
test_array_times = [200, 100, 50, 35, 25,  15,  10,   7]

test_arrays = [init_array_fixed(n) for n in test_array_sizes]
test_functions = [
    kernel_cholesky_slicing,
    kernel_cholesky_torchapi,
    kernel_cholesky_banachiewicz_for,
    kernel_cholesky_banachiewicz_nofor,
    kernel_cholesky_banachiewicz_nofor_optim,
    kernel_cholesky_algorithm_w1,
    kernel_cholesky_crout
]

total_runtimes = {key.__name__:{} for key in test_functions}

for func in test_functions:
    for i in range(len(test_array_sizes)):
        runtimes = []
        for t in range(test_array_times[i]):
            # Run the function on the test array
            A = test_arrays[i].clone()
            start = perf_counter()
            func(test_array_sizes[i], A)
            runtimes.append(perf_counter() - start)
        total_runtimes[func.__name__][test_array_sizes[i]] = sum(runtimes) / len(runtimes)
            
        print(f"\x1b[34maverage for \x1b[33m{func.__name__} \x1b[36m(size {test_array_sizes[i]}): \x1b[0m{round(sum(runtimes)*1000 / len(runtimes), 3)}ms")
    print("\n+-------------------------------------------------------+\n")
    
with open("runtimes.txt", "w") as f:
    for funcname in total_runtimes.keys():
        f.write(f"{funcname}\n")
        for size in total_runtimes[funcname].keys():
            f.write(f"{round(1000*total_runtimes[funcname][size], 3)}\n")
        f.write("\n")