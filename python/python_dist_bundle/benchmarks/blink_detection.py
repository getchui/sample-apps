from observation import Observation
from typing import List
from utils import (Parameters, MemoryHighWaterMarkTracker, Stopwatch, SDKFactory)
import tfsdk


_benchmark_name = 'Blink detection'


def benchmark(gpu_options: tfsdk.GPUOptions, parameters: Parameters, observations: List[Observation]) -> None:
    mem_tracker = MemoryHighWaterMarkTracker()

    # Initialize the SDK
    sdk = SDKFactory.createSDK(
        gpu_options,
        initialize_modules=['face_detector', 'blink_detector'])

    # Load the image
    ret, img = sdk.preprocess_image('./headshot.jpg')
    if ret != tfsdk.ERRORCODE.NO_ERROR:
        print('Error: could not load the image')
        return

    ret, found, face_box_and_landmarks = sdk.detect_largest_face(img)
    if ret != tfsdk.ERRORCODE.NO_ERROR or found is False:
        print('Unable to detect face in image')
        return

    if parameters.do_warmup:
        for _ in range(parameters.num_warmup):
            error_code, blink_state = sdk.detect_blink(img, face_box_and_landmarks)
            if error_code != tfsdk.ERRORCODE.NO_ERROR:
                print('Error: Unable to run blink detection')
                return

    # Time the blink detector
    times = []
    for _ in range(parameters.num_iterations):
        stop_watch = Stopwatch()
        sdk.detect_blink(img, face_box_and_landmarks)
        times.append(stop_watch.elapsedTime())

    observations.append(
        Observation(
            sdk.get_version(), gpu_options.enable_GPU,
            _benchmark_name, '',
            parameters, times, mem_tracker.get_diff_from_baseline()))
