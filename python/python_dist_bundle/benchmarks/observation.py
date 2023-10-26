import csv
from dataclasses import dataclass
import os
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


class ObservationCSVWriter:
    _fieldNames = [
        'SDK Version',
        'GPU or CPU',
        'Benchmark Name',
        'Benchmark Type or Model',
        'Batch Size',
        'Number of Iterations',
        'Total Time (ms)',
        'Mean Time (ms)',
        'Variance (ms)',
        'Low (ms)',
        'High (ms)'
    ]

    def __init__(self, path: str) -> None:
        self.path = path

    def write(self, observations: List[Observation]) -> None:
        write_header = False if os.path.exists(self.path) else True

        with open(self.path, 'a') as csvfile:
            writer = csv.writer(csvfile)

            if write_header is True:
                writer.writerow(ObservationCSVWriter._fieldNames)

            for o in observations:
                writer.writerow([
                    o.version,
                    'GPU' if o.is_gpu_enabled else 'CPU',
                    o.benchmark_name,
                    o.benchmark_subtype,
                    o.params.batch_size,
                    o.params.num_iterations,
                    f'{o.time.total:0.4f}',
                    f'{o.time.mean:0.4f}',
                    f'{o.time.variance:0.4f}',
                    f'{o.time.low:0.4f}',
                    f'{o.time.high:0.4f}'
                ])
