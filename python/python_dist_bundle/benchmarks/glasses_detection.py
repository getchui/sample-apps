from observation import Observation
from typing import List
from utils import (Parameters, Stopwatch, SDKFactory)
import tfsdk


_benchmark_name = 'Eyeglasses detection'


def benchmark(gpu_options: tfsdk.GPUOptions, parameters: Parameters, observations: List[Observation]) -> None:
    # Initialize the SDK
    sdk = SDKFactory.createSDK(gpu_options, initialize_modules=['eyeglass_detector'])

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
            error_code, glasses_label, glasses_score = \
                sdk.detect_glasses(img, face_box_and_landmarks)
            if error_code != tfsdk.ERRORCODE.NO_ERROR:
                print('Error: Unable to run glasses detection')
                return

    # Time the glasses detector
    times = []
    for _ in range(parameters.num_iterations):
        stop_watch = Stopwatch()
        sdk.detect_glasses(img, face_box_and_landmarks)
        times.append(stop_watch.elapsedTime())

    observations.append(
        Observation(sdk.get_version(), gpu_options.enable_GPU, _benchmark_name, '', parameters, times))
