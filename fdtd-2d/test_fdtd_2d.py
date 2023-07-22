from fdtd_2d_POLY import (
    kernel_fdtd_2d_original,
    kernel_fdtd_2d_some_optimization,
    kernel_fdtd_2d_no_forloop,
    kernel_fdtd_2d_diff,
    kernel_fdtd_2d_index_shift,
    kernel_fdtd_2d_roll,
    init_array
)

RESET = '\033[0m'
COLOR_CODE_DIM = "\x1B[30m" # 30: Gray
COLOR_CODE_RED = "\x1B[31m" # 31: Red
COLOR_CODE_GREEN = "\x1B[32m" # 32: Green
COLOR_CODE_YELLOW = "\x1B[33m" # 33: Yellow
COLOR_CODE_BLUE = "\x1B[34m" # 34: Blue
COLOR_CODE_MAGENTA = "\x1B[35m" # 35: Magenta
COLOR_CODE_CYAN = "\x1B[36m" # 36: Cyan
COLOR_CODE_WHITE = "\x1B[37m" # 37: White
COLOR_CODE_NORMAL = "\x1B[0m" # 0: Normal
COLOR_CODE_BOLD = "\x1B[1m" # 1: Bold
COLOR_CODE_DIM = "\x1B[2m" # 2: Dim
COLOR_CODE_ITALIC = "\x1B[3m" # 3: Italic
COLOR_CODE_UNDERLINE = "\x1B[4m" # 4: Underlined
COLOR_CODE_BLINKING = "\x1B[5m" # 5: Blinking
COLOR_CODE_FASTBLINKING = "\x1B[6m" # 6: Fast Blinking
COLOR_CODE_REVERSE = "\x1B[7m" # 7: Reverse
COLOR_CODE_INVISIBLE = "\x1B[8m" # 8: Invisible

import torch
from copy import deepcopy
import colorama
import traceback

PRESETS = {
    'debug': (2, 2, 2),
    'mini': (2, 32, 32),
    'small': (10, 500, 500),
    'standard': (50, 1000, 1000),
    'large': (50, 2000, 2000),
    'extralarge': (100, 4000, 4000)
}

DATA_TYPE = torch.float64

dataset_size = input('Problem size: ')
if (dataset_size not in PRESETS):
    print(COLOR_CODE_DIM + "Unrecognized problem size. Enter custom parameters: " + RESET)
    TMAX = int(input('TMAX (Timesteps): '))
    NX = int(input('NX (x-dim of mesh): '))
    NY = int(input('NY (y-dim of mesh): '))
    print()
else: 
    TMAX, NX, NY = PRESETS[dataset_size]
    print(COLOR_CODE_DIM + f"\nUsing preset parameters: TMAX: {TMAX}, NX: {NX}, NY: {NY}" + RESET)

def format_2d_list(li: list) -> str:
    """
    Formats a 2D list into a string
    Meant to be used with small lists
    """
    
    # Find longest row in string form
    longest_str_len = max([len(" ".join([str(round(item, 3)) for item in row])) for row in li]) 
    rowspacer = "| " + ' '*longest_str_len + " |\n"
    horiz_line = "+" + "-"*(longest_str_len+2) + "+\n"
    
    # Find the longest number in each column. Format the string so that each column is the same length
    longest_nums_in_cols = []
    for col in range(len(li[0])):
        longest_num_in_col = max([len(str(round(row[col], 3))) for row in li])
        longest_nums_in_cols.append(longest_num_in_col)
      
    row_strs = [] 
    
    for row in range(len(li)):
        # use max length of each column to format the string
        
        raw_spaced_numbers_str = ""
        for num in range(len(li[row]) - 1):
            num_spaces_needed = longest_nums_in_cols[li[row].index(li[row][num])] - len(str(round(li[row][num], 3)))
            front_spaces = " "*(num_spaces_needed//2)
            back_spaces = " "*(num_spaces_needed - len(front_spaces))
            raw_spaced_numbers_str += front_spaces + str(round(li[row][num], 3)) + back_spaces + " "
        raw_spaced_numbers_str += str(round(li[row][-1], 3))
        
        row_strs.append("| " + raw_spaced_numbers_str + " "*(longest_str_len - len(raw_spaced_numbers_str)) + " |\n")

    return horiz_line + rowspacer.join(row_strs) + horiz_line


# Variable declaration/allocation.
# In python, array declarations are not needed. Just pass the array as an argument
# Keep in mind, in the original code, NX = nx and NY = ny
nx = NX
ny = NY
tmax = TMAX
print(COLOR_CODE_DIM + "Creating arrays..." + RESET)

# POLYBENCH_2D_ARRAY_DECL(ex,DATA_TYPE,NX,NY,nx,ny); for ex, ey, hz
ex = ey = hz = torch.zeros((nx, ny), dtype=DATA_TYPE)

#POLYBENCH_1D_ARRAY_DECL(_fict_,DATA_TYPE,TMAX,tmax);
_fict_ = torch.zeros(tmax, dtype=DATA_TYPE)
print(COLOR_CODE_DIM + "Initalizing values...\n" + RESET)

# Initialize array(s).
# POLYBENCH_ARRAY returns the pointer in C, but in python, we just return the array
ex, ey, hz, _fict_ = init_array(tmax, nx, ny, DATA_TYPE)

# _fict_ is not modified in the kernel, so we can just use the same array

TEST_AGAINST = kernel_fdtd_2d_original
TEST_FUNCTIONS: list = [
    kernel_fdtd_2d_some_optimization,
    kernel_fdtd_2d_no_forloop,
    kernel_fdtd_2d_diff,
    kernel_fdtd_2d_roll,
    kernel_fdtd_2d_index_shift,
]

print(colorama.Fore.CYAN + "Running original kernel..." + colorama.Style.RESET_ALL)
accepted_value: tuple = TEST_AGAINST(tmax, nx, ny, deepcopy(ex), deepcopy(ey), deepcopy(hz), _fict_)

return_values = []
for func in TEST_FUNCTIONS:
    print(colorama.Fore.CYAN + f"Testing {func.__name__}..." + colorama.Style.RESET_ALL)
    #print("Params: \n")
    #print(f"tmax: {tmax}; nx: {nx}; ny: {ny}")
    #print(f"ex: {deepcopy(ex).tolist()}, ey: {deepcopy(ey).tolist()}, hz: {deepcopy(hz).tolist()}")
    #print(f"_fict_: {_fict_.tolist()}")
    try:
        return_values.append( # Should return a tuple of the three modified tensors
            func(
                tmax, nx, ny, 
                deepcopy(ex), 
                deepcopy(ey), 
                deepcopy(hz), _fict_
            )
        )
    except Exception as e:
        print(colorama.Fore.RED + f"\nError in {func.__name__}: {e}\n" + colorama.Style.RESET_ALL)
        traceback.print_exc()
        print('\n')
        continue
        
    
# Assert that all return values are equal

var_names = ['ex', 'ey', 'hz']

for result_index in range(len(return_values)):
    printstr = f"Results for function {TEST_FUNCTIONS[result_index].__name__}..."
    print("="*len(printstr)+"\n"+printstr+"\n"+"="*len(printstr))
    all_equal = True
    for i in range(len(accepted_value)): # there should be 3 tensors in the tuple: ex, ey, hz
        # Float inaccuracy
        if not torch.allclose(accepted_value[i], return_values[result_index][i], rtol=1e-03, atol=1e-03):
            all_equal = False
            print(colorama.Fore.YELLOW + f"\x1B[4m{var_names[i]} is incorrect." + colorama.Style.RESET_ALL)
            print(colorama.Fore.GREEN + f"Accepted value:\n{format_2d_list(accepted_value[i].tolist())}" + colorama.Style.RESET_ALL)
            print(colorama.Fore.RED + f"Result value:\n{format_2d_list(return_values[result_index][i].tolist())}\n" + colorama.Style.RESET_ALL)
    if all_equal:
        print(colorama.Fore.GREEN + "All values are correct!" + colorama.Style.RESET_ALL)
    print('\n')