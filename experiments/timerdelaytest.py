import datetime
from time import sleep
import statistics

def get_stats(sleep_time_seconds: float, iterations: int) -> tuple:
    times = []
    for i in range(iterations):
        start = datetime.datetime.now()
        sleep(sleep_time_seconds)
        end = datetime.datetime.now()
        times.append((end-start).microseconds-(sleep_time_seconds*1000000))
    avg = statistics.mean(times)
    min_time = min(times)
    max_time = max(times)
    med = statistics.median(times)
    iqr_25 = statistics.quantiles(times, n=4)[0]
    iqr_75 = statistics.quantiles(times, n=4)[2]
    return sleep_time_seconds, avg, min_time, iqr_25, med, iqr_75, max_time

def __write_stats_tuple(data: tuple, filepath: str):
    with open(filepath, "a") as f:
        f.write(','.join([str(x) for x in data]) + '\n')
        
for ms_sleeptime in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000]:
    sleeptime = ms_sleeptime / 1000
    iterations = int(10 / (sleeptime + 0.01*int(sleeptime == 0)))  # keey at 10 seconds per test, but don't divide by 0
    print(f"Iterating {min(1000, iterations)} times with {sleeptime} seconds of sleep")
    data = get_stats(sleeptime, iterations=min(1000, iterations))
    __write_stats_tuple(data, "stats.csv")