from utils import (Parameters, Stopwatch, SDKFactory)
import tfsdk


def benchmark(gpu_options: tfsdk.GPUOptions, parameters: Parameters) -> None:
    # Initialize the SDK
    sdk = SDKFactory.createSDK(gpu_options, initialize_modules=['face_detector'])

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

    chips = parameters.batch_size * [chip]

    if parameters.do_warmup:
        for _ in range(parameters.num_warmup):
            error_code, mask_labels, mask_scores = sdk.detect_masks(chips)
            if error_code != tfsdk.ERRORCODE.NO_ERROR:
                print('Error: Unable to run mask detection')
                return

    # Time the mask detector
    stop_watch = Stopwatch()
    for _ in range(parameters.num_iterations):
        sdk.detect_masks(chips)
    total_time = stop_watch.elapsedTimeMilliSeconds()
    avg_time = total_time / parameters.num_iterations / parameters.batch_size

    print("Average time mask detection: {} ms | batch size = {} | {} iterations".format(
        avg_time, parameters.batch_size, parameters.num_iterations))
