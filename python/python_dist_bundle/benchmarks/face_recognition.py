from observation import Observation
from utils import (Parameters, Stopwatch, SDKFactory)
import tfsdk


_benchmark_name = 'Face recognition'


def benchmark(fr_model: tfsdk.FACIALRECOGNITIONMODEL, gpu_options: tfsdk.GPUOptions, parameters: Parameters) -> None:
    # Initialize the SDK
    sdk = SDKFactory.createSDK(
        gpu_options,
        initialize_modules=['face_recognizer'],
        fr_model=fr_model)

    # Load the image
    ret, img = sdk.preprocess_image('./headshot.jpg')
    if ret != tfsdk.ERRORCODE.NO_ERROR:
        print('Error: could not load the image')
        return

    ret, found, face_box_and_landmarks = sdk.detect_largest_face(img)
    if ret != tfsdk.ERRORCODE.NO_ERROR or found is False:
        print('Error: Unable to detect face when benchmarking face recognition model')
        return

    ret, chip = sdk.extract_aligned_face(img, face_box_and_landmarks)
    if ret != tfsdk.ERRORCODE.NO_ERROR:
        print('Error: Unable to extract aligned face for mask detection')
        return

    chips = parameters.batch_size * [chip]

    if parameters.do_warmup:
        for _ in range(parameters.num_warmup):
            error_code, faceprints = sdk.get_face_feature_vectors(chips)
            if error_code != tfsdk.ERRORCODE.NO_ERROR:
                print('Error: Unable to run face recognition')
                return

    times = []
    for _ in range(parameters.num_iterations):
        stop_watch = Stopwatch()
        sdk.get_face_feature_vectors(chips)
        times.append(stop_watch.elapsedTime())

    o = Observation(sdk.get_version(), gpu_options.enable_GPU, _benchmark_name, fr_model.name, parameters, times)
