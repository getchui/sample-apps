from stopwatch import Stopwatch
import tfsdk

import os

NUM_WARMUP = 10
DO_WARMUP = True


def benchmark(license, gpu_options, batch_size, num_iterations):
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

    # Load the image
    ret, img = sdk.preprocess_image("./headshot.jpg")
    if ret != tfsdk.ERRORCODE.NO_ERROR:
        print('Error: could not load the image')
        return

    ret, found, face_box_and_landmarks = sdk.detect_largest_face(img)
    if ret != tfsdk.ERRORCODE.NO_ERROR or found is False:
        print('Unable to detect face in image')
        return

    ret, chip = sdk.extract_aligned_face(img, face_box_and_landmarks)
    if ret != tfsdk.ERRORCODE.NO_ERROR:
        print('Error: Unable to extract aligned face for mask detection')
        return

    chips = batch_size*[chip]

    if DO_WARMUP:
        for _ in range(NUM_WARMUP):
            error_code, mask_labels, mask_scores = sdk.detect_masks(chips)
            if error_code != tfsdk.ERRORCODE.NO_ERROR:
                print('Error: Unable to run mask detection')
                return

    # Time the mask detector
    stop_watch = Stopwatch()
    for _ in range(num_iterations):
        sdk.detect_masks(chips)
    total_time = stop_watch.elapsedTimeMilliSeconds()
    avg_time = total_time / num_iterations / batch_size

    print("Average time mask detection: {} ms | batch size = {} | {} iterations".format(
        avg_time, batch_size, num_iterations))
