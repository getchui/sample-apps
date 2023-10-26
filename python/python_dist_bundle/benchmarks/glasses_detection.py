from utils import (Parameters, Stopwatch, SDKFactory)
import tfsdk


def benchmark(gpu_options: tfsdk.GPUOptions, parameters: Parameters) -> None:
    # Initialize the SDK
    sdk = SDKFactory.createSDK(gpu_options, initialize_modules=['eyeglass_detector'])

    # Load the image
    ret, img = sdk.preprocess_image("./headshot.jpg")
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
    stop_watch = Stopwatch()
    for _ in range(parameters.num_iterations):
        sdk.detect_glasses(img, face_box_and_landmarks)
    total_time = stop_watch.elapsedTimeMilliSeconds()
    avg_time = total_time / parameters.num_iterations

    print('Average time glasses detection:',
          avg_time, 'ms |', parameters.num_iterations, 'iterations')
