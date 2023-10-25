from stopwatch import Stopwatch
import tfsdk

import os

NUM_WARMUP = 10
DO_WARMUP = True


def benchmark(license, gpu_options, num_iterations = 100):
    # Initialize the SDK
    options = tfsdk.ConfigurationOptions()

    options.models_path = "./"
    models_path = os.getenv('MODELS_PATH')
    if models_path:
        options.models_path = models_path

    options.GPU_options = gpu_options

    initialize_module = tfsdk.InitializeModule()
    initialize_module.face_detector = True
    initialize_module.landmark_detector = True
    options.initialize_module = initialize_module

    sdk = tfsdk.SDK(options)

    is_valid = sdk.set_license(license)
    if is_valid is False:
        print('Error: the provided license is invalid.')
        exit(1)

    ret, img = sdk.preprocess_image("./headshot.jpg")
    if ret != tfsdk.ERRORCODE.NO_ERROR:
        print('Error: could not load the image')
        return

    ret, found, face_box_and_landmarks = sdk.detect_largest_face(img)
    if ret != tfsdk.ERRORCODE.NO_ERROR or found is False:
        print('Unable to detect face in image')
        return

    if DO_WARMUP:
        for _ in range(NUM_WARMUP):
            error_code, landmarks = sdk.get_face_landmarks(img, face_box_and_landmarks)
            if error_code != tfsdk.ERRORCODE.NO_ERROR:
                print('Error: Unable to get detailed landmarks')
                return

    # Time the landmark detection
    stop_watch = Stopwatch()
    for i in range(num_iterations):
        sdk.get_face_landmarks(img, face_box_and_landmarks)
    total_time = stop_watch.elapsedTimeMilliSeconds()
    avg_time = total_time / num_iterations

    print("Average time 106 face landmark detection: {} ms | {} iterations".format(
        avg_time, num_iterations))
