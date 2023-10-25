from utils import (Parameters, Stopwatch)
import tfsdk

import os


def benchmark(license: str, gpu_options: tfsdk.GPUOptions,
              obj_model: tfsdk.OBJECTDETECTIONMODEL, parameters: Parameters) -> None:
    # Initialize the SDK
    options = tfsdk.ConfigurationOptions()

    options.models_path = "./"
    models_path = os.getenv('MODELS_PATH')
    if models_path:
        options.models_path = models_path

    options.GPU_options = gpu_options
    options.obj_model = obj_model

    initialize_module = tfsdk.InitializeModule()
    initialize_module.object_detector = True
    options.initialize_module = initialize_module

    sdk = tfsdk.SDK(options)

    is_valid = sdk.set_license(license)
    if is_valid is False:
        print('Error: the provided license is invalid.')
        exit(1)

    # Load the image
    ret, img = sdk.preprocess_image("./headshot.jpg")
    if ret != tfsdk.ERRORCODE.NO_ERROR:
        print('Error: could not load the image')
        return

    if parameters.do_warmup:
        for _ in range(parameters.num_warmup):
            ret, objects = sdk.detect_objects(img)
            if ret != tfsdk.ERRORCODE.NO_ERROR:
                print('Error: Could not run object detection!')
                return

    # Time the creation of the feature vector
    stop_watch = Stopwatch()
    for _ in range(parameters.num_iterations):
        sdk.detect_objects(img)
    total_time = stop_watch.elapsedTimeMilliSeconds()
    avg_time = total_time / parameters.num_iterations

    print('Average time object detection ({}): {} ms | {} iterations'.format(
        obj_model.name, avg_time, parameters.num_iterations))
