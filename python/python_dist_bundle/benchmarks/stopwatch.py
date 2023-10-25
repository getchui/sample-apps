import time

class Stopwatch:
    def __init__(self):
        self.start_point = time.time_ns()

    def elapsedTime(self):
        return time.time_ns() - self.start_point

    def elapsedTimeMilliSeconds(self):
        now = time.time_ns()
        return now / (10 ** 6) - self.start_point / (10**6)