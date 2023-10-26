from utils import (Parameters, Stopwatch, SDKFactory)
import tfsdk


def benchmark(gpu_options: tfsdk.GPUOptions,
              obj_model: tfsdk.OBJECTDETECTIONMODEL,
              parameters: Parameters) -> None:
    # Initialize the SDK
    sdk = SDKFactory.createSDK(
        gpu_options,
        initialize_modules=['object_detector'],
        obj_model=obj_model)

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
