from observation import Observation
from typing import List
from utils import (Parameters, Stopwatch, SDKFactory)
import tfsdk


_benchmark_name = 'Face image orientation detection'


def benchmark(gpu_options: tfsdk.GPUOptions, parameters: Parameters, observations: List[Observation]) -> None:
    # Initialize the SDK
    sdk = SDKFactory.createSDK(gpu_options, initialize_modules=[])

    img_path = './headshot.jpg'
    # Load the image
    error_code, img = sdk.preprocess_image(img_path)
    if error_code != tfsdk.ERRORCODE.NO_ERROR:
        print('Error: could not load the image')
        return

    if parameters.do_warmup:
        for _ in range(parameters.num_warmup):
            error_code, flags = sdk.get_face_image_rotation(img)
            if error_code != tfsdk.ERRORCODE.NO_ERROR:
                print('Error: Unable to compute face image orientation')
                return

    times = []
    for _ in range(parameters.num_iterations):
        stop_watch = Stopwatch()
        sdk.get_face_image_rotation(img)
        times.append(stop_watch.elapsedTime())

    observations.append(
        Observation(sdk.get_version(), gpu_options.enable_GPU, _benchmark_name, '', parameters, times))
