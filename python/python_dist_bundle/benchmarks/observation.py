from dataclasses import dataclass
import statistics
from typing import List
import utils


@dataclass
class TimeResult:
    total: float
    mean: float
    variance: float
    low: float
    high: float


@dataclass
class Observation:
    def __init__(self, version: str, is_gpu_enabled: bool, benchmark_name: str, benchmark_subtype: str, params: utils.Parameters, times: List[float]):
        self.version = version
        self.is_gpu_enabled = is_gpu_enabled
        self.benchmark_name = benchmark_name
        self.benchmark_subtype = benchmark_subtype
        self.params = params
        self.time = self._summarize_times(params, times)
        print(self)

    def _summarize_times(self, params: utils.Parameters, times: List[float]) -> TimeResult:
        normalized_times = list(map(lambda x: x / params.batch_size, times))

        time_result = TimeResult(
            total=sum(normalized_times) / utils.Stopwatch.ns_in_ms,
            mean=statistics.fmean(normalized_times) / utils.Stopwatch.ns_in_ms,
            low=min(normalized_times) / utils.Stopwatch.ns_in_ms,
            high=max(normalized_times) / utils.Stopwatch.ns_in_ms,
            variance=statistics.variance(normalized_times) / utils.Stopwatch.ns_in_ms)

        return time_result

    def __str__(self) -> str:
        s = f'Average time {self.benchmark_name}'
        if self.benchmark_subtype:
            s += f' ({self.benchmark_subtype})'
        s += f': {self.time.mean: 0.3f} ms'

        if self.params.batch_size > 1:
            s += f' | batch size = {self.params.batch_size}'

        s += f' | {self.params.num_iterations} iterations'

        return s
