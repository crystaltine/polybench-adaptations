from time import perf_counter

from lu_POLY import (
    init_array,
    kernel_lu_original,
    kernel_lu_arange_calc_TEST,
    kernel_lu_nofor,
    kernel_lu_slicing,
    kernel_lu_torchapi,
    kernel_lu_tril,
)

# Run each function on larger and larger inputs
test_array_sizes = [5,   10,  15, 20, 25, 30, 50, 100]
test_array_times = [200, 100, 50, 30, 20, 10,  7,   3]

test_arrays = [init_array(n) for n in test_array_sizes]
test_functions = [
    kernel_lu_arange_calc_TEST,
    kernel_lu_nofor,
    kernel_lu_slicing,
    kernel_lu_torchapi,
    kernel_lu_tril,
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