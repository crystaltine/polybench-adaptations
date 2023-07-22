import time
from unused.progress_bar import COLOR_CODE_YELLOW, RESET, get_color_escape, BOLD

def run_function(__func, *args, **kwargs):
    """
    Runs and times a function using Python's built-in time.perf_counter().
    """
    start = time.perf_counter()
    __func(*args, **kwargs)
    end = time.perf_counter()
    
    print(f"Function {COLOR_CODE_YELLOW + __func.__name__ + RESET} completed in {get_color_escape(255, 0, 255) + BOLD + str(round(1000 * (end - start)))}ms" + RESET)