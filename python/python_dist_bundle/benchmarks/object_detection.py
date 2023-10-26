from observation import Observation
from utils import (Parameters, Stopwatch, SDKFactory)
import tfsdk


_benchmark_name = 'Object detection'


def benchmark(gpu_options: tfsdk.GPUOptions,
              obj_model: tfsdk.OBJECTDETECTIONMODEL,
              parameters: Parameters) -> None:
    # Initialize the SDK
    sdk = SDKFactory.createSDK(
        gpu_options,
        initialize_modules=['object_detector'],
        obj_model=obj_model)

    # Load the image
    ret, img = sdk.preprocess_image('./headshot.jpg')
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
    times = []
    for _ in range(parameters.num_iterations):
        stop_watch = Stopwatch()
        sdk.detect_objects(img)
        times.append(stop_watch.elapsedTime())

    o = Observation(sdk.get_version(), gpu_options.enable_GPU, _benchmark_name, obj_model.name, parameters, times)
