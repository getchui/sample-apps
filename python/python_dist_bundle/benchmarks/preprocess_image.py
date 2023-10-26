from observation import Observation
from typing import List
from utils import (Parameters, Stopwatch, SDKFactory)
import tfsdk

import os


_benchmark_name = 'Preprocess image'


def benchmark(gpu_options: tfsdk.GPUOptions, parameters: Parameters, observations: List[Observation]) -> None:
    # Initialize the SDK
    sdk = SDKFactory.createSDK(gpu_options, initialize_modules=[])

    img_path = './headshot.jpg'

    # First run the benchmark for an image on disk
    # Run once to ensure everything works
    if parameters.do_warmup:
        for _ in range(parameters.num_warmup):
            error_code, img = sdk.preprocess_image(img_path)
            if error_code != tfsdk.ERRORCODE.NO_ERROR:
                print('Unable to preprocess the image')
                return

    # Time the preprocess_image function
    times = []
    for _ in range(parameters.num_iterations):
        stop_watch = Stopwatch()
        sdk.preprocess_image(img_path)
        times.append(stop_watch.elapsedTime())

    observations.append(
        Observation(sdk.get_version(), gpu_options.enable_GPU, _benchmark_name, 'JPG from disk', parameters, times))

    # Now repeat with encoded image in memory
    size = os.path.getsize(img_path)
    with open(img_path, 'rb') as infile:
        data = infile.read(size)

    buffer = []
    for b in data:
        buffer.append(b)

    if parameters.do_warmup:
        for _ in range(parameters.num_warmup):
            error_code, img = sdk.preprocess_image(buffer)
            if error_code != tfsdk.ERRORCODE.NO_ERROR:
                print('Unable to preprocess the image')
                return

    # Time the preprocess_image function
    times = []
    for _ in range(parameters.num_iterations):
        stop_watch = Stopwatch()
        sdk.preprocess_image(buffer)
        times.append(stop_watch.elapsedTime())

    observations.append(
        Observation(sdk.get_version(), gpu_options.enable_GPU, _benchmark_name, 'encoded JPG in memory', parameters, times))

    # Now repeat with already decoded images (ex. you grab an image from your video stream).
    if parameters.do_warmup:
        for _ in range(parameters.num_warmup):
            error_code, img = sdk.preprocess_image(img.get_data(), img.get_width(), img.get_height(), tfsdk.COLORCODE.rgb)
            if error_code != tfsdk.ERRORCODE.NO_ERROR:
                print('Error: Unable to preprocess image')
                return

    times = []
    for _ in range(parameters.num_iterations):
        stop_watch = Stopwatch()
        sdk.preprocess_image(img.get_data(), img.get_width(), img.get_height(), tfsdk.COLORCODE.rgb)
        times.append(stop_watch.elapsedTime())

    observations.append(
        Observation(sdk.get_version(), gpu_options.enable_GPU, _benchmark_name, 'RGB pixels array in memory', parameters, times))
