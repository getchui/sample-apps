from stopwatch import Stopwatch
import tfsdk

import os

NUM_WARMUP = 10
DO_WARMUP = True

def benchmark(license, gpu_options, num_iterations = 200):
    # Initialize the SDK
    options = tfsdk.ConfigurationOptions()

    options.models_path = "./"
    models_path = os.getenv('MODELS_PATH')
    if models_path:
        options.models_path = models_path

    options.GPU_options = gpu_options

    initialize_module = tfsdk.InitializeModule()
    initialize_module.face_detector = True
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

    ret, landmarks = sdk.get_face_landmarks(img, face_box_and_landmarks)
    if ret != tfsdk.ERRORCODE.NO_ERROR:
        print('Unable to detect landmarks')
        return

    if DO_WARMUP:
        for _ in range(NUM_WARMUP):
            error_code, yaw, pitch, roll, rotation_vec, translation_vec = \
                sdk.estimate_head_orientation(img, face_box_and_landmarks, landmarks)

    # Time the head orientation
    stop_watch = Stopwatch()
    for _ in range(num_iterations):
        sdk.estimate_head_orientation(img, face_box_and_landmarks, landmarks)
    total_time = stop_watch.elapsedTimeMilliSeconds()
    avg_time = total_time / num_iterations

    print("Average time head orientation: {} ms | {} iterations".format(avg_time, num_iterations))
