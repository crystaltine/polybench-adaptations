COLOR_CODE_GRAY = "\x1B[30m" # 30: Gray
COLOR_CODE_RED = "\x1B[31m" # 31: Red
COLOR_CODE_GREEN = "\x1B[32m" # 32: Green
COLOR_CODE_YELLOW = "\x1B[33m" # 33: Yellow
COLOR_CODE_BLUE = "\x1B[34m" # 34: Blue
COLOR_CODE_MAGENTA = "\x1B[35m" # 35: Magenta
COLOR_CODE_CYAN = "\x1B[36m" # 36: Cyan
COLOR_CODE_WHITE = "\x1B[37m" # 37: White
RESET = '\033[0m'

BOLD = '\033[1m'

def get_color_escape(r, g, b, background=False):
    return '\033[{};2;{};{};{}m'.format(48 if background else 38, r, g, b)

def _get_progress_string(progress: float, length: int = 20) -> str:
    block = "█"
    
    num_blocks = int(round(progress * length))
    num_empty = length - num_blocks
    
    if num_blocks == length: # so that there is no extra half block if the progress is 100%
        return "[" + block * num_blocks + "]"
    
    return "[" + block * num_blocks + get_color_escape(70, 76, 86) + block * num_empty + RESET + "]"
