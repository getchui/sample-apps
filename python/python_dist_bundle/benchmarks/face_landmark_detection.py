from utils import (Parameters, Stopwatch, SDKFactory)
import tfsdk


def benchmark(gpu_options: tfsdk.GPUOptions, parameters: Parameters) -> None:
    # Initialize the SDK
    sdk = SDKFactory.createSDK(gpu_options, initialize_modules=['face_detector'])

    ret, img = sdk.preprocess_image("./headshot.jpg")
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
    stop_watch = Stopwatch()
    for _ in range(parameters.num_iterations):
        sdk.detect_largest_face(img)
    total_time = stop_watch.elapsedTimeMilliSeconds()
    avg_time = total_time / parameters.num_iterations

    print("Average time face and landmark detection: {} ms | {} iterations".format(
        avg_time, parameters.num_iterations))
