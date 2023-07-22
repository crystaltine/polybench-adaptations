from jacobi_2d_imper_POLY import (
    jacobi_2d_imper_original,
    jacobi_2d_imper_conv,
    jacobi_2d_imper_imp3,
    jacobi_2d_imper_imp4,
    jacobi_2d_imper_imp5,
    init_array_2d
)

RESET = '\033[0m'
COLOR_CODE_BLACK = "\x1B[30m" # 30: Gray
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
    'debug': (2, 3),
    'mini': (2, 32),
    'small': (10, 500),
    'standard': (20, 1000),
    'large': (20, 2000),
    'extralarge': (100, 4000)
}

DATA_TYPE = torch.float64

dataset_size = input('Problem size: ')
if (dataset_size not in PRESETS):
    print(COLOR_CODE_DIM + "Unrecognized problem size. Enter custom parameters: " + RESET)
    TMAX = int(input('TMAX (Timesteps): '))
    N = int(input('N (side length of mesh): '))
    print()
else: 
    TMAX, N = PRESETS[dataset_size]
    print(COLOR_CODE_DIM + f"\nUsing preset parameters: TMAX: {TMAX}, N: {N}" + RESET)

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
            raw_spaced_numbers_str += str(round(li[row][num], 3)) + " "*num_spaces_needed + " "
        raw_spaced_numbers_str += str(round(li[row][-1], 3))
        
        row_strs.append("| " + raw_spaced_numbers_str + " "*(longest_str_len - len(raw_spaced_numbers_str)) + " |\n")

    return horiz_line + rowspacer.join(row_strs) + horiz_line

print(COLOR_CODE_DIM + "Initalizing tensor...\n" + RESET)
initial = init_array_2d(N)

TEST_AGAINST = jacobi_2d_imper_original
TEST_FUNCTIONS: list = [
    jacobi_2d_imper_conv,
    jacobi_2d_imper_imp3,
    jacobi_2d_imper_imp4,
    jacobi_2d_imper_imp5
]

print(colorama.Fore.CYAN + "Running original calculation..." + colorama.Style.RESET_ALL)
accepted_value = TEST_AGAINST(TMAX, deepcopy(initial))

return_values = []
for func in TEST_FUNCTIONS:
    print(colorama.Fore.CYAN + f"Testing {func.__name__}..." + colorama.Style.RESET_ALL)
    #print("Params: \n")
    #print(f"tmax: {tmax}; nx: {nx}; ny: {ny}")
    #print(f"ex: {deepcopy(ex).tolist()}, ey: {deepcopy(ey).tolist()}, hz: {deepcopy(hz).tolist()}")
    #print(f"_fict_: {_fict_.tolist()}")
    try:
        return_values.append( # Should return a tuple of the three modified tensors
            func(TMAX, deepcopy(initial))
        )
    except Exception as e:
        return_values.append(None)
        print(colorama.Fore.RED + f"\nError in {func.__name__}: {e}\n" + colorama.Style.RESET_ALL)
        traceback.print_exc()
        print('\n')
        continue
        
    
# Assert that all return values are equal
for result_index in range(len(return_values)):
    
    printstr = f"Results for function {TEST_FUNCTIONS[result_index].__name__}..."
    print("="*len(printstr)+"\n"+printstr+"\n"+"="*len(printstr))
    
    # if exception then skip
    if return_values[result_index] is None:
        print(colorama.Fore.YELLOW + f"\x1B[4mTest case failed." + colorama.Style.RESET_ALL)
        print(colorama.Fore.RED + f"See error above.\n" + colorama.Style.RESET_ALL)
    # Float inaccuracy
    elif not torch.allclose(accepted_value, return_values[result_index], rtol=1e-03, atol=1e-03):
        print(colorama.Fore.YELLOW + f"\x1B[4mTest case failed." + colorama.Style.RESET_ALL)
        print(colorama.Fore.GREEN + f"Accepted value:\n{format_2d_list(accepted_value.tolist())}" + colorama.Style.RESET_ALL)
        print(colorama.Fore.RED + f"Result value:\n{format_2d_list(return_values[result_index].tolist())}\n" + colorama.Style.RESET_ALL)
    else:
        print(colorama.Fore.GREEN + "Test case passed!" + colorama.Style.RESET_ALL)
    print('\n')