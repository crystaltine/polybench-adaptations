from lu_POLY import (
    init_array,
    kernel_lu_original,
    kernel_lu_slicing,
    kernel_lu_tril,
    kernel_lu_arange_calc,
    kernel_lu_torchapi
)

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
    'debug': 3,
    'mini': 40,
    'small': 120,
    'standard': 400,
    'large': 2000,
    'extralarge': 4000
}

DATA_TYPE = torch.float64

dataset_size = input('Problem size: ')
if (dataset_size not in PRESETS):
    print(COLOR_CODE_DIM + "Unrecognized problem size. Enter custom parameters: " + RESET)
    N = int(input('N (side length of matrix to calculate): '))
    print()
else: 
    N = PRESETS[dataset_size]
    print(COLOR_CODE_DIM + f"\nUsing preset parameters: N={N}" + RESET)

print(COLOR_CODE_DIM + "Creating arrays..." + RESET)

ORIGINAL_A = init_array(N)

TEST_AGAINST = kernel_lu_original
TEST_FUNCTIONS: list = [
    kernel_lu_slicing,
    kernel_lu_arange_calc,
    kernel_lu_torchapi,
    kernel_lu_tril
]

print(colorama.Fore.CYAN + "Running original kernel..." + colorama.Style.RESET_ALL)

accepted_value = deepcopy(ORIGINAL_A)
TEST_AGAINST(N, accepted_value)

return_values = []
for func in TEST_FUNCTIONS:
    print(colorama.Fore.CYAN + f"Testing {func.__name__}..." + colorama.Style.RESET_ALL)
    try:
        A = deepcopy(ORIGINAL_A)
        func(N, A)
        return_values.append(A)
    except Exception as e:
        return_values.append(None)
        print(colorama.Fore.RED + f"\nError in {func.__name__}: {e}\n" + colorama.Style.RESET_ALL)
        traceback.print_exc()
        print('\n')
        continue
        
    
for result_index in range(len(return_values)):
    printstr = f"Results for function {TEST_FUNCTIONS[result_index].__name__}..."
    print("="*len(printstr)+"\n"+printstr+"\n"+"="*len(printstr))
    
    # if exception then skip
    if return_values[result_index] is None:
        print(colorama.Fore.YELLOW + f"\x1B[4mTest case failed: Error." + colorama.Style.RESET_ALL)
        print(colorama.Fore.RED + f"See error above.\n" + colorama.Style.RESET_ALL)
    # Float inaccuracy
    elif not torch.allclose(accepted_value, return_values[result_index], rtol=1e-03, atol=1e-03):
        print(colorama.Fore.YELLOW + f"\x1B[4mTest case failed: Incorrect." + colorama.Style.RESET_ALL)
        print(colorama.Fore.GREEN + f"Accepted value:\n{format_2d_list(accepted_value.tolist())}" + colorama.Style.RESET_ALL)
        print(colorama.Fore.RED + f"Result value:\n{format_2d_list(return_values[result_index].tolist())}\n" + colorama.Style.RESET_ALL)
    else:
        print(colorama.Fore.GREEN + "Test case passed!" + colorama.Style.RESET_ALL)
    print('\n')