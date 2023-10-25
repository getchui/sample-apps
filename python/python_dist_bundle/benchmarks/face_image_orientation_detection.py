from utils import (Parameters, Stopwatch)

import tfsdk

import os

NUM_WARMUP = 10
DO_WARMUP = True

def benchmark(license: str, gpu_options: tfsdk.GPUOptions, parameters: Parameters) -> None:
    # Initialize the SDK
    options = tfsdk.ConfigurationOptions()

    options.models_path = "./"
    models_path = os.getenv('MODELS_PATH')
    if models_path:
        options.models_path = models_path

    options.GPU_options = gpu_options

    sdk = tfsdk.SDK(options)

    is_valid = sdk.set_license(license)
    if (is_valid is False):
        print('Error: the provided license is invalid.')
        exit(1)

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

    stop_watch = Stopwatch()
    for _ in range(parameters.num_iterations):
        sdk.get_face_image_rotation(img)
    total_time = stop_watch.elapsedTimeMilliSeconds()
    avg_time = total_time / parameters.num_iterations

    print('Average time face image orientation detection: {} ms  | {} iterations'.format(
        avg_time, parameters.num_iterations))
