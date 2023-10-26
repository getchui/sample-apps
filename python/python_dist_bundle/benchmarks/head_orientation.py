from observation import Observation
from typing import List
from utils import (Parameters, Stopwatch, SDKFactory)
import tfsdk


_benchmark_name = 'Head orientation detection (yaw, pitch, roll)'


def benchmark(gpu_options: tfsdk.GPUOptions, parameters: Parameters, observations: List[Observation]) -> None:
    # Initialize the SDK
    sdk = SDKFactory.createSDK(gpu_options, initialize_modules=['face_detector'])

    ret, img = sdk.preprocess_image('./headshot.jpg')
    if ret != tfsdk.ERRORCODE.NO_ERROR:
        print('Error: could not load the image')
        return

    ret, found, face_box_and_landmarks = sdk.detect_largest_face(img)
    if ret != tfsdk.ERRORCODE.NO_ERROR or found is False:
        print('Unable to detect face in image')
        return

    ret, landmarks = sdk.get_face_landmarks(img, face_box_and_landmarks)
    if ret != tfsdk.ERRORCODE.NO_ERROR:
        print('Unable to detect landmarks')
        return

    if parameters.do_warmup:
        for _ in range(parameters.num_warmup):
            error_code, yaw, pitch, roll, rotation_vec, translation_vec = \
                sdk.estimate_head_orientation(img, face_box_and_landmarks, landmarks)

    # Time the head orientation
    times = []
    for _ in range(parameters.num_iterations):
        stop_watch = Stopwatch()
        sdk.estimate_head_orientation(img, face_box_and_landmarks, landmarks)
        times.append(stop_watch.elapsedTime())

    observations.append(
        Observation(sdk.get_version(), gpu_options.enable_GPU, _benchmark_name, '', parameters, times))
