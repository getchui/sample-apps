from dataclasses import dataclass
import time


@dataclass
class Parameters:
    do_warmup: bool = True
    num_warmup: int = 10
    batch_size: int = 16
    num_iterations: int = 200


class Stopwatch:
    def __init__(self):
        self.start_point = time.time_ns()

    def elapsedTime(self):
        return time.time_ns() - self.start_point

    def elapsedTimeMilliSeconds(self):
        now = time.time_ns()
        return now / (10 ** 6) - self.start_point / (10**6)
