from convolution_3d_POLY import (
    init_array,
    kernel_conv3d_direct_translation,
    kernel_conv3d_torchconv,
    kernel_conv3d_roll
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
    'debug': (3, 3, 3),
    'mini': (64, 64, 64),
    'small': (128, 128, 128),
    'standard': (192, 192, 192),
    'large': (256, 256, 256),
    'extralarge': (384, 384, 384)
}

DATA_TYPE = torch.float64

dataset_size = input('Problem size: ')
if (dataset_size not in PRESETS):
    print(COLOR_CODE_DIM + "Unrecognized problem size. Enter custom parameters: " + RESET)
    NI = int(input('NI (x-dim of tensor): '))
    NJ = int(input('NJ (y-dim of tensor): '))
    NK = int(input('NK (z-dim of tensor): '))
    print()
else: 
    NI, NJ, NK = PRESETS[dataset_size]
    print(COLOR_CODE_DIM + f"\nUsing preset parameters: NI={NI}, NJ={NJ}, NK={NK}\n" + RESET)

print(COLOR_CODE_DIM + "Creating arrays..." + RESET)

A = init_array(NI, NJ, NK)

TEST_AGAINST = kernel_conv3d_direct_translation
TEST_FUNCTIONS: list = [
    kernel_conv3d_torchconv,
    kernel_conv3d_roll
]

print(colorama.Fore.CYAN + "Running original kernel..." + colorama.Style.RESET_ALL)
accepted_value: tuple = TEST_AGAINST(NI, NJ, NK, deepcopy(A))

return_values = []
for func in TEST_FUNCTIONS:
    print(colorama.Fore.CYAN + f"Testing {func.__name__}..." + colorama.Style.RESET_ALL)
    try:
        return_values.append(func(NI, NJ, NK, deepcopy(A)))
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
        print(colorama.Fore.YELLOW + f"\x1B[4mTest case failed." + colorama.Style.RESET_ALL)
        print(colorama.Fore.RED + f"See error above.\n" + colorama.Style.RESET_ALL)
    # Float inaccuracy
    elif not torch.allclose(accepted_value, return_values[result_index], rtol=1e-03, atol=1e-03):
        print(colorama.Fore.YELLOW + f"\x1B[4mTest case failed." + colorama.Style.RESET_ALL)
        print(colorama.Fore.GREEN + f"Accepted value:\n{accepted_value}" + colorama.Style.RESET_ALL)
        print(colorama.Fore.RED + f"Result value:\n{return_values[result_index]}\n" + colorama.Style.RESET_ALL)
    else:
        print(colorama.Fore.GREEN + "Test case passed!" + colorama.Style.RESET_ALL)
    print('\n')