# Run benchmarks

import tfsdk
import os
import time

NUM_WARMUP = 10
DO_WARMUP = True

class Stopwatch:
    def __init__(self):
        self.start_point = time.time_ns()

    def elapsedTime(self):
        return time.time_ns() - self.start_point

    def elapsedTimeMilliSeconds(self):
        now = time.time_ns()
        return now / (10 ** 6) - self.start_point / (10**6)


def benchmark_preprocess_image(license, gpu_options, num_iterations = 200):
    # Initialize the SDK
    options = tfsdk.ConfigurationOptions()

    options.models_path = "./"
    models_path = os.getenv('MODELS_PATH')
    if models_path:
        options.models_path = models_path

    options.GPU_options = gpu_options

    sdk = tfsdk.SDK(options)

    is_valid = sdk.set_license(license)
    if (is_valid is False):
        print('Error: the provided license is invalid.')
        exit(1)

    img_path = './headshot.jpg'

    # First run the benchmark for an image on disk
    # Run once to ensure everything works
    if DO_WARMUP:
        for _ in range(NUM_WARMUP):
            error_code, img = sdk.preprocess_image(img_path)
            if error_code != tfsdk.ERRORCODE.NO_ERROR:
                print("Unable to preprocess the image")
                return

    # Time the preprocess_image function
    stop_watch = Stopwatch()
    for _ in range(num_iterations):
        sdk.preprocess_image(img_path)
    total_time = stop_watch.elapsedTimeMilliSeconds()
    avg_time = total_time / num_iterations

    print("Average time preprocess_image JPG image from disk ({}x{}): {} ms | {} iterations".format(
        img.get_width(), img.get_height(), avg_time, num_iterations))

    # Now repeat with encoded image in memory
    size = os.path.getsize(img_path)
    with open(img_path, 'rb') as infile:
        data = infile.read(size)

    buffer = []
    for b in data:
        buffer.append(b)

    if DO_WARMUP:
        for _ in range(NUM_WARMUP):
            error_code, img = sdk.preprocess_image(buffer)
            if error_code != tfsdk.ERRORCODE.NO_ERROR:
                print("Unable to preprocess the image")
                return

    # Time the preprocess_image function
    stop_watch = Stopwatch()
    for _ in range(num_iterations):
        sdk.preprocess_image(buffer)
    total_time = stop_watch.elapsedTimeMilliSeconds()
    avg_time = total_time / num_iterations

    print("Average time preprocess_image JPG image in memory ({}x{}): {} ms | {} iterations".format(
        img.get_width(), img.get_height(), avg_time, num_iterations))


def benchmark_face_image_orientation_detection(license, gpu_options, num_iterations):
    # Initialize the SDK
    options = tfsdk.ConfigurationOptions()

    options.models_path = "./"
    models_path = os.getenv('MODELS_PATH')
    if models_path:
        options.models_path = models_path

    options.GPU_options = gpu_options

    sdk = tfsdk.SDK(options)

    is_valid = sdk.set_license(license)
    if (is_valid is False):
        print('Error: the provided license is invalid.')
        exit(1)

    img_path = './headshot.jpg'
    # Load the image
    error_code, img = sdk.preprocess_image(img_path)
    if error_code != tfsdk.ERRORCODE.NO_ERROR:
        print('Error: could not load the image')
        return

    if DO_WARMUP:
        for _ in range(NUM_WARMUP):
            error_code, flags = sdk.get_face_image_rotation(img)
            if error_code != tfsdk.ERRORCODE.NO_ERROR:
                print('Error: Unable to compute face image orientation')
                return

    stop_watch = Stopwatch()
    for _ in range(num_iterations):
        sdk.get_face_image_rotation(img)
    total_time = stop_watch.elapsedTimeMilliSeconds()
    avg_time = total_time / num_iterations

    print('Average time face image orientation detection: {} ms  | {} iterations'.format(
        avg_time, num_iterations))


def benchmark_face_landmark_detection(license, gpu_options, num_iterations = 100):
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

    if DO_WARMUP:
        for _ in range(NUM_WARMUP):
            found, face_box_and_landmarks = sdk.detect_largest_face(img)

    # Time the face detection
    stop_watch = Stopwatch()
    for _ in range(num_iterations):
        sdk.detect_largest_face(img)
    total_time = stop_watch.elapsedTimeMilliSeconds()
    avg_time = total_time / num_iterations

    print("Average time face and landmark detection: {} ms | {} iterations".format(
        avg_time, num_iterations))


def benchmark_detailed_landmark_detection(license, gpu_options, num_iterations = 100):
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

    found, face_box_and_landmarks = sdk.detect_largest_face(img)
    if found is False:
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


def benchmark_blink_detection(license, gpu_options, num_iterations = 100):
    # Initialize the SDK
    options = tfsdk.ConfigurationOptions()

    options.models_path = "./"
    models_path = os.getenv('MODELS_PATH')
    if models_path:
        options.models_path = models_path

    options.GPU_options = gpu_options

    initialize_module = tfsdk.InitializeModule()
    initialize_module.face_detector = True
    initialize_module.blink_detector = True
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

    found, face_box_and_landmarks = sdk.detect_largest_face(img)
    if found is False:
        print('Unable to detect face in image')
        return

    if DO_WARMUP:
        for _ in range(NUM_WARMUP):
            error_code, blink_state = sdk.detect_blink(img, face_box_and_landmarks)
            if error_code != tfsdk.ERRORCODE.NO_ERROR:
                print('Error: Unable to run blink detection')
                return

    # Time the blink detector
    stop_watch = Stopwatch()
    for _ in range(num_iterations):
        sdk.detect_blink(img, face_box_and_landmarks)
    total_time = stop_watch.elapsedTimeMilliSeconds()
    avg_time = total_time / num_iterations

    print("Average time blink detection: {} ms | {} iterations".format(avg_time, num_iterations))


def benchmark_spoof_detection(license, gpu_options, num_iterations = 100):
    # Initialize the SDK
    options = tfsdk.ConfigurationOptions()

    options.models_path = "./"
    models_path = os.getenv('MODELS_PATH')
    if models_path:
        options.models_path = models_path

    options.GPU_options = gpu_options

    initialize_module = tfsdk.InitializeModule()
    initialize_module.face_detector = True
    initialize_module.passive_spoof = True
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

    found, face_box_and_landmarks = sdk.detect_largest_face(img)
    if found is False:
        print('Unable to detect face in image')
        return

    if DO_WARMUP:
        for _ in range(NUM_WARMUP):
            error_code, spoof_label, spoof_score = sdk.detect_spoof(img, face_box_and_landmarks)
            if error_code != tfsdk.ERRORCODE.NO_ERROR:
                print('Spoof function failed')
                print(error_code)
                return

    # Time the spoof detector
    stop_watch = Stopwatch()
    for _ in range(num_iterations):
        sdk.detect_spoof(img, face_box_and_landmarks)
    total_time = stop_watch.elapsedTimeMilliSeconds()
    avg_time = total_time / num_iterations

    print("Average time spoof detection: {} ms | {} iterations".format(avg_time, num_iterations))


def benchmark_mask_detection(license, gpu_options, batch_size, num_iterations):
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

    found, face_box_and_landmarks = sdk.detect_largest_face(img)
    if found is False:
        print('Unable to detect face in image')
        return

    chip = sdk.extract_aligned_face(img, face_box_and_landmarks)
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


def benchmark_glasses_detection(license, gpu_options, num_iterations):
    # Initialize the SDK
    options = tfsdk.ConfigurationOptions()

    options.models_path = "./"
    models_path = os.getenv('MODELS_PATH')
    if models_path:
        options.models_path = models_path

    options.GPU_options = gpu_options

    initialize_module = tfsdk.InitializeModule()
    initialize_module.eyeglass_detector = True
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

    found, face_box_and_landmarks = sdk.detect_largest_face(img)
    if found is False:
        print('Unable to detect face in image')
        return

    if DO_WARMUP:
        for _ in range(NUM_WARMUP):
            error_code, glasses_label, glasses_score = \
                sdk.detect_glasses(img, face_box_and_landmarks)
            if error_code != tfsdk.ERRORCODE.NO_ERROR:
                print('Error: Unable to run glasses detection')
                return

    # Time the glasses detector
    stop_watch = Stopwatch()
    for _ in range(num_iterations):
        sdk.detect_glasses(img, face_box_and_landmarks)
    total_time = stop_watch.elapsedTimeMilliSeconds()
    avg_time = total_time / num_iterations

    print('Average time glasses detection:',
          avg_time, 'ms |', num_iterations, 'iterations')


def benchmark_head_orientation(license, gpu_options, num_iterations = 200):
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

    found, face_box_and_landmarks = sdk.detect_largest_face(img)
    if found is False:
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


def benchmark_face_image_blur_detection(license, gpu_options, num_iterations):
    # Initialize the SDK
    options = tfsdk.ConfigurationOptions()

    options.models_path = "./"
    models_path = os.getenv('MODELS_PATH')
    if models_path:
        options.models_path = models_path

    options.GPU_options = gpu_options

    initialize_module = tfsdk.InitializeModule()
    initialize_module.face_blur_detector = True
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

    found, face_box_and_landmarks = sdk.detect_largest_face(img)
    if found is False:
        print('Unable to detect face in image')
        return

    face_chip = sdk.extract_aligned_face(img, face_box_and_landmarks)

    if DO_WARMUP:
        for _ in range(NUM_WARMUP):
            error_code, quality, score = sdk.detect_face_image_blur(face_chip)
            if error_code != tfsdk.ERRORCODE.NO_ERROR:
                print('Error: Unable to detect face image blur')
                return

    # Time the mask detector
    stop_watch = Stopwatch()
    for _ in range(num_iterations):
        sdk.detect_face_image_blur(face_chip)
    total_time = stop_watch.elapsedTimeMilliSeconds()
    avg_time = total_time / num_iterations

    print('Average time face image blur detection:',
          avg_time, 'ms  |', num_iterations, 'iterations')


def benchmark_object_detection(license, gpu_options, obj_model, num_iterations = 100):
    # Initialize the SDK
    options = tfsdk.ConfigurationOptions()

    options.models_path = "./"
    models_path = os.getenv('MODELS_PATH')
    if models_path:
        options.models_path = models_path

    options.GPU_options = gpu_options
    options.obj_model = obj_model

    initialize_module = tfsdk.InitializeModule()
    initialize_module.object_detector = True
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

    if DO_WARMUP:
        for _ in range(NUM_WARMUP):
            objects = sdk.detect_objects(img)

    # Time the creation of the feature vector
    stop_watch = Stopwatch()
    for _ in range(num_iterations):
        sdk.detect_objects(img)
    total_time = stop_watch.elapsedTimeMilliSeconds()
    avg_time = total_time / num_iterations

    print('Average time object detection ({}): {} ms | {} iterations'.format(
        obj_model.name, avg_time, num_iterations))


def benchmark_face_recognition(license, fr_model, gpu_options, batch_size = 1, num_iterations = 100):
    # Initialize the SDK
    options = tfsdk.ConfigurationOptions()

    options.models_path = "./"
    models_path = os.getenv('MODELS_PATH')
    if models_path:
        options.models_path = models_path

    options.GPU_options = gpu_options
    options.fr_model = fr_model

    initialize_module = tfsdk.InitializeModule()
    initialize_module.face_recognizer = True
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

    found, face_box_and_landmarks = sdk.detect_largest_face(img)
    if found is False:
        print('Error: Unable to detect face when benchmarking face recognition model')
        return

    chip = sdk.extract_aligned_face(img, face_box_and_landmarks)
    chips = batch_size*[chip]

    if DO_WARMUP:
        for _ in range(NUM_WARMUP):
            error_code, faceprints = sdk.get_face_feature_vectors(chips)
            if error_code != tfsdk.ERRORCODE.NO_ERROR:
                print('Error: Unable to run face recognition')
                return

    stop_watch = Stopwatch()
    for _ in range(num_iterations):
        sdk.get_face_feature_vectors(chips)
    total_time = stop_watch.elapsedTimeMilliSeconds()
    avg_time = total_time / num_iterations / batch_size

    print("Average time face recognition {}: {} ms | batch size = {} | {} iterations".format(
        fr_model.name, avg_time, batch_size, num_iterations))


# ********************************************************************************************************
# ********************************************************************************************************
#
#                                  START OF BENCHMARKING SCRIPT
#
# ********************************************************************************************************
# ********************************************************************************************************

def main():
    license = os.environ['TRUEFACE_TOKEN']

    gpu_options = tfsdk.GPUOptions()
    gpu_options.enable_GPU = False # TODO: Set this to true to benchmark on GPU.
    gpu_options.device_index = 0

    gpu_module_options = tfsdk.GPUModuleOptions()
    gpu_module_options.precision = tfsdk.PRECISION.FP16

    batch_size = 16
    gpu_module_options.max_batch_size = batch_size
    gpu_module_options.opt_batch_size = 1

    gpu_options.face_detector_GPU_options = gpu_module_options
    gpu_options.face_recognizer_GPU_options = gpu_module_options
    gpu_options.mask_detector_GPU_options = gpu_module_options
    gpu_options.object_detector_GPU_options = gpu_module_options
    gpu_options.face_landmark_detector_GPU_options = gpu_module_options
    gpu_options.face_orientation_detector_GPU_options = gpu_module_options
    gpu_options.face_blur_detector_GPU_options = gpu_module_options
    gpu_options.spoof_detector_GPU_options = gpu_module_options
    gpu_options.blink_detector_GPU_options = gpu_module_options

    print("==========================")
    print("==========================")
    if gpu_options.enable_GPU is True:
        print("Using GPU for inference")
    else:
        print("Using CPU for inference")
    print("==========================")
    print("==========================")

    mult_factor = 1
    if gpu_options.enable_GPU is True:
        mult_factor = 10

    benchmark_preprocess_image(license, gpu_options, 200)
    benchmark_face_image_orientation_detection(license, gpu_options, 50*mult_factor)
    benchmark_face_landmark_detection(license, gpu_options, 100*mult_factor)
    benchmark_detailed_landmark_detection(license, gpu_options, 100*mult_factor)
    benchmark_head_orientation(license, gpu_options, 500*mult_factor)
    benchmark_face_image_blur_detection(license, gpu_options, 200*mult_factor)
    benchmark_blink_detection(license, gpu_options, 100*mult_factor)
    benchmark_mask_detection(license, gpu_options, 1, 100*mult_factor)
    benchmark_glasses_detection(license, gpu_options, 200)
    benchmark_spoof_detection(license, gpu_options, 100*mult_factor)
    benchmark_object_detection(license, gpu_options, tfsdk.OBJECTDETECTIONMODEL.FAST, 100*mult_factor)
    benchmark_object_detection(license, gpu_options, tfsdk.OBJECTDETECTIONMODEL.ACCURATE, 40*mult_factor)

    if gpu_options.enable_GPU is False:
        # get_face_feature_vectors is not supported by the LITE model
        benchmark_face_recognition(license, tfsdk.FACIALRECOGNITIONMODEL.LITE, gpu_options, 1, 200)

    benchmark_face_recognition(license, tfsdk.FACIALRECOGNITIONMODEL.LITE_V2, gpu_options, 1, 200)
    benchmark_face_recognition(license, tfsdk.FACIALRECOGNITIONMODEL.TFV5_2, gpu_options, 1, 40*mult_factor)
    benchmark_face_recognition(license, tfsdk.FACIALRECOGNITIONMODEL.TFV6, gpu_options, 1, 40*mult_factor)
    benchmark_face_recognition(license, tfsdk.FACIALRECOGNITIONMODEL.TFV7, gpu_options, 1, 40*mult_factor)
    # Benchmarks with batching.
    # On CPU, should be the same speed as a batch size of 1.
    # On GPU, will increase the throughput.
    benchmark_face_recognition(license, tfsdk.FACIALRECOGNITIONMODEL.TFV5_2, gpu_options, batch_size, 40*mult_factor)
    benchmark_face_recognition(license, tfsdk.FACIALRECOGNITIONMODEL.TFV6, gpu_options, batch_size, 40*mult_factor)
    benchmark_face_recognition(license, tfsdk.FACIALRECOGNITIONMODEL.TFV7, gpu_options, batch_size, 40*mult_factor)

    benchmark_mask_detection(license, gpu_options, batch_size, 100*mult_factor)


if __name__ == '__main__':
    main()
