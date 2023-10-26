from observation import Observation
from typing import List
from utils import (Parameters, Stopwatch, SDKFactory)
import tfsdk


_benchmark_name = 'Face and landmark detection'


def benchmark(gpu_options: tfsdk.GPUOptions, parameters: Parameters, observations: List[Observation]) -> None:
    # Initialize the SDK
    sdk = SDKFactory.createSDK(gpu_options, initialize_modules=['face_detector'])

    ret, img = sdk.preprocess_image('./headshot.jpg')
    if ret != tfsdk.ERRORCODE.NO_ERROR:
        print('Error: could not load the image')
        return

    if parameters.do_warmup:
        for _ in range(parameters.num_warmup):
            ret, found, face_box_and_landmarks = sdk.detect_largest_face(img)
            if ret != tfsdk.ERRORCODE.NO_ERROR:
                print('Error: Unable to run face detection')
                return

    # Time the face detection
    times = []
    for _ in range(parameters.num_iterations):
        stop_watch = Stopwatch()
        sdk.detect_largest_face(img)
        times.append(stop_watch.elapsedTime())

    observations.append(
        Observation(sdk.get_version(), gpu_options.enable_GPU, _benchmark_name, '', parameters, times))
