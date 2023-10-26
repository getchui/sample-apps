from utils import (Parameters, Stopwatch, SDKFactory)
import tfsdk


def benchmark(gpu_options: tfsdk.GPUOptions, parameters: Parameters) -> None:
    # Initialize the SDK
    sdk = SDKFactory.createSDK(gpu_options, initialize_modules=['face_detector'])

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

    if parameters.do_warmup:
        for _ in range(parameters.num_warmup):
            error_code, yaw, pitch, roll, rotation_vec, translation_vec = \
                sdk.estimate_head_orientation(img, face_box_and_landmarks, landmarks)

    # Time the head orientation
    stop_watch = Stopwatch()
    for _ in range(parameters.num_iterations):
        sdk.estimate_head_orientation(img, face_box_and_landmarks, landmarks)
    total_time = stop_watch.elapsedTimeMilliSeconds()
    avg_time = total_time / parameters.num_iterations

    print("Average time head orientation: {} ms | {} iterations".format(avg_time, parameters.num_iterations))
