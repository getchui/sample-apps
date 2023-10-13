# Run benchmarks

import tfsdk
import os
from colorama import Fore
from colorama import Style
from time import time
import inspect

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


def current_milli_time():
    return round(time.time() * 1000)

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

    # Now repeat with already decoded imgages (ex. you grab an image from your video stream).
    # @todo SDK-235 img.get_data is not implemented. i think this will require a
    # wrapper around the pointer, but i haven't fully grokked pybind11.
    return

    if DO_WARMUP:
        for i in range(10):
            error_code = sdk.preprocess_image(
                img.get_data(), img.get_width(), img.get_height(),
                tfsdk.COLORCODE.rgb)
            if error_code != tfsdk.ERRORCODE.NO_ERROR:
                print("Unable to preprocess the image")
                return

    stop_watch = Stopwatch()
    for _ in range(num_iterations):
        sdk.preprocess_image(
                buffer, img.get_width(), img.get_height(), tfsdk.COLORCODE.bgr)
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
            #
            # @todo: SDK-235 pybinding does not return an error code, should we add
            #        to match others in the SDK?
            #
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

    #
    # @todo: SDK-235 pybinding does not return an error code, should we add
    #        to match others in the SDK?
    #
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
    options = tfsdk.ConfigurationOptions()
    options.models_path = os.getenv('MODELS_PATH') or './'
    options.GPU_options = gpu_options

    options.smallest_face_height = 40
    options.initialize_module.face_detector = True
    options.initialize_module.blink_detector = True

    sdk = tfsdk.SDK(options)

    is_valid = sdk.set_license(os.environ['TRUEFACE_TOKEN'])
    if (is_valid == False):
        print(f"{Fore.RED}Invalid License Provided{Style.RESET_ALL}")
        print(f"{Fore.RED}Be sure to export your license token as TRUEFACE_TOKEN{Style.RESET_ALL}")
        quit()


    ret, img = sdk.preprocess_image("./headshot.jpg")
    if (ret != tfsdk.ERRORCODE.NO_ERROR):
        print("There was an error setting the image in the {} method".format(inspect.stack()[0][3]))
        quit()

    found, fb = sdk.detect_largest_face(img)
    if found == False:
        print("Unable to find face in {} method".format(inspect.stack()[0][3]))
        quit()

    # Run our timing code
    t1 = current_milli_time()
    for i in range(num_iterations):
        ret, blink_state = sdk.detect_blink(img, fb)
    t2 = current_milli_time()

    if ret != tfsdk.ERRORCODE.NO_ERROR:
        print("Unable to run blink detection in {} method".format(inspect.stack()[0][3]))

    total_time = t2 - t1
    avg_time = total_time / num_iterations

    print("Average time blink detection: {} ms | {} iterations".format(avg_time, num_iterations))

def benchmark_spoof_detection(license, gpu_options, num_iterations = 100):
    options = tfsdk.ConfigurationOptions()
    options.models_path = os.getenv('MODELS_PATH') or './'
    options.GPU_options = gpu_options

    options.smallest_face_height = 40
    options.initialize_module.face_detector = True
    options.initialize_module.passive_spoof = True

    sdk = tfsdk.SDK(options)

    is_valid = sdk.set_license(os.environ['TRUEFACE_TOKEN'])
    if (is_valid == False):
        print(f"{Fore.RED}Invalid License Provided{Style.RESET_ALL}")
        print(f"{Fore.RED}Be sure to export your license token as TRUEFACE_TOKEN{Style.RESET_ALL}")
        quit()


    ret, img = sdk.preprocess_image("./real_spoof.jpg")
    if (ret != tfsdk.ERRORCODE.NO_ERROR):
        print("There was an error setting the image in the {} method".format(inspect.stack()[0][3]))
        quit()

    found, fb = sdk.detect_largest_face(img)
    if found == False:
        print("Unable to find face in {} method".format(inspect.stack()[0][3]))
        quit()

    # Run our timing code
    t1 = current_milli_time()
    for i in range(num_iterations):
        ret, spoof_label, spoof_score = sdk.detect_spoof(img, fb)
    t2 = current_milli_time()

    if ret != tfsdk.ERRORCODE.NO_ERROR:
        print("Unable to run spoof detection in {} method".format(inspect.stack()[0][3]))

    total_time = t2 - t1
    avg_time = total_time / num_iterations

    print("Average time spoof detection: {} ms | {} iterations".format(avg_time, num_iterations))


def benchmark_mask_detection(license, gpu_options, batch_size = 1, num_iterations = 100):
    options = tfsdk.ConfigurationOptions()
    options.models_path = os.getenv('MODELS_PATH') or './'
    options.GPU_options = gpu_options

    options.smallest_face_height = 40
    options.initialize_module.face_detector = True

    sdk = tfsdk.SDK(options)

    is_valid = sdk.set_license(os.environ['TRUEFACE_TOKEN'])
    if (is_valid == False):
        print(f"{Fore.RED}Invalid License Provided{Style.RESET_ALL}")
        print(f"{Fore.RED}Be sure to export your license token as TRUEFACE_TOKEN{Style.RESET_ALL}")
        quit()


    ret, img = sdk.preprocess_image("./headshot.jpg")
    if (ret != tfsdk.ERRORCODE.NO_ERROR):
        print("There was an error setting the image in the {} method".format(inspect.stack()[0][3]))
        quit()

    found, fb = sdk.detect_largest_face(img)
    if found == False:
        print("Unable to find face in {} method".format(inspect.stack()[0][3]))
        quit()

    chip = sdk.extract_aligned_face(img, fb)
    chips = []

    for i in range(batch_size):
        chips.append(chip)

    # Run our timing code
    t1 = current_milli_time()
    for i in range(num_iterations):
        ret, mask_labels, mask_score = sdk.detect_masks(chips)
    t2 = current_milli_time()

    if ret != tfsdk.ERRORCODE.NO_ERROR:
        print("Unable to run mask detection in {} method".format(inspect.stack()[0][3]))

    total_time = t2 - t1
    avg_time = total_time / num_iterations / batch_size

    print("Average time mask detection: {} ms | batch size = {} | {} iterations".format(avg_time, batch_size, num_iterations))

def benchmark_head_orientation(license, gpu_options, num_iterations = 200):
    options = tfsdk.ConfigurationOptions()
    options.models_path = os.getenv('MODELS_PATH') or './'
    options.GPU_options = gpu_options

    options.smallest_face_height = 40
    options.initialize_module.face_detector = True

    sdk = tfsdk.SDK(options)

    is_valid = sdk.set_license(os.environ['TRUEFACE_TOKEN'])
    if (is_valid == False):
        print(f"{Fore.RED}Invalid License Provided{Style.RESET_ALL}")
        print(f"{Fore.RED}Be sure to export your license token as TRUEFACE_TOKEN{Style.RESET_ALL}")
        quit()


    ret, img = sdk.preprocess_image("./headshot.jpg")
    if (ret != tfsdk.ERRORCODE.NO_ERROR):
        print("There was an error setting the image in the {} method".format(inspect.stack()[0][3]))
        quit()

    found, fb = sdk.detect_largest_face(img)
    if found == False:
        print("Unable to find face in {} method".format(inspect.stack()[0][3]))
        quit()

    ret, landmarks = sdk.get_face_landmarks(img, fb)
    if (ret != tfsdk.ERRORCODE.NO_ERROR):
        print("Unable to get face landmarks in {} method".format(inspect.stack()[0][3]))
        quit()

    # Run our timing code
    t1 = current_milli_time()
    for i in range(num_iterations):
        ret, yaw, pitch, roll, rotation_vec, translation_vec = sdk.estimate_head_orientation(img, fb, landmarks)
    t2 = current_milli_time()

    if ret != tfsdk.ERRORCODE.NO_ERROR:
        print("Unable to estimate head orientation in {} method".format(inspect.stack()[0][3]))

    total_time = t2 - t1
    avg_time = total_time / num_iterations

    print("Average time head orientation: {} ms | {} iterations".format(avg_time, num_iterations))


def benchmark_object_detection(license, gpu_options, num_iterations = 100):
    options = tfsdk.ConfigurationOptions()
    options.models_path = os.getenv('MODELS_PATH') or './'
    options.GPU_options = gpu_options

    options.obj_model = tfsdk.OBJECTDETECTIONMODEL.FAST
    options.initialize_module.object_detector = True
    options.GPU_options = gpu_options

    sdk = tfsdk.SDK(options)

    is_valid = sdk.set_license(os.environ['TRUEFACE_TOKEN'])
    if (is_valid == False):
        print(f"{Fore.RED}Invalid License Provided{Style.RESET_ALL}")
        print(f"{Fore.RED}Be sure to export your license token as TRUEFACE_TOKEN{Style.RESET_ALL}")
        quit()


    ret, img = sdk.preprocess_image("./bike.jpg")
    if (ret != tfsdk.ERRORCODE.NO_ERROR):
        print("There was an error setting the image in the {} method".format(inspect.stack()[0][3]))
        quit()

    # Run our timing code
    t1 = current_milli_time()
    for i in range(num_iterations):
        objects = sdk.detect_objects(img)
    t2 = current_milli_time()

    total_time = t2 - t1
    avg_time = total_time / num_iterations

    print("Average time object detection ({}x{}): {} ms | {} iterations".format(img.get_width(), img.get_height(), avg_time, num_iterations))


def benchmark_face_recognition(license, fr_model, gpu_options, batch_size = 1, num_iterations = 100):
    options = tfsdk.ConfigurationOptions()
    options.models_path = os.getenv('MODELS_PATH') or './'
    options.GPU_options = gpu_options
    options.fr_model = fr_model

    options.initialize_module.face_recognizer = True

    sdk = tfsdk.SDK(options)

    is_valid = sdk.set_license(os.environ['TRUEFACE_TOKEN'])
    if (is_valid == False):
        print(f"{Fore.RED}Invalid License Provided{Style.RESET_ALL}")
        print(f"{Fore.RED}Be sure to export your license token as TRUEFACE_TOKEN{Style.RESET_ALL}")
        quit()


    ret, img = sdk.preprocess_image("./headshot.jpg")
    if (ret != tfsdk.ERRORCODE.NO_ERROR):
        print("There was an error setting the image in the {} method".format(inspect.stack()[0][3]))
        quit()

    found, fb = sdk.detect_largest_face(img)
    if found == False:
        print("Unable to find face in {} method".format(inspect.stack()[0][3]))
        quit()

    chip = sdk.extract_aligned_face(img, fb)
    chips = []

    for i in range(batch_size):
        chips.append(chip)

    # Run our timing code
    t1 = current_milli_time()
    for i in range(num_iterations):
        ret, faceprints = sdk.get_face_feature_vectors(chips)
    t2 = current_milli_time()

    if ret != tfsdk.ERRORCODE.NO_ERROR:
        print("Unable to run face recognition in {} method".format(inspect.stack()[0][3]))

    total_time = t2 - t1
    avg_time = total_time / num_iterations / batch_size

    print("Average time face recognition {}: {} ms | batch size = {} | {} iterations".format(fr_model.name, avg_time, batch_size, num_iterations))

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
    gpu_options.enable_GPU = True # TODO: Set this to true to benchmark on GPU.
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

if __name__ == '__main__':
    main()


# benchmark_blink_detection(license, gpu_options)
# benchmark_spoof_detection(license, gpu_options)
# benchmark_mask_detection(license, gpu_options, 1, 100 * mult_factor)
# benchmark_head_orientation(license, gpu_options)
# benchmark_object_detection(license, gpu_options, 100 * mult_factor)

# if gpu_options.enable_GPU == False:
#     # get_face_feature_vectors method is not support by the LITE and LITE_V2 models
#     benchmark_face_recognition(license, tfsdk.FACIALRECOGNITIONMODEL.LITE, gpu_options, 1, 200)
#     benchmark_face_recognition(license, tfsdk.FACIALRECOGNITIONMODEL.LITE_V2, gpu_options, 1, 200)

# benchmark_face_recognition(license, tfsdk.FACIALRECOGNITIONMODEL.TFV7, gpu_options, 1, 40 * mult_factor)
# benchmark_face_recognition(license, tfsdk.FACIALRECOGNITIONMODEL.TFV6, gpu_options, 1, 40 * mult_factor)
# benchmark_face_recognition(license, tfsdk.FACIALRECOGNITIONMODEL.TFV5_2, gpu_options, 1, 40 * mult_factor)

# # Benchmarks with batching.
# # On CPU, should be the same speed as a batch size of 1.
# # On GPU, will increase the throughput.
# benchmark_face_recognition(license, tfsdk.FACIALRECOGNITIONMODEL.TFV7, gpu_options, batch_size, 40 * mult_factor)
# benchmark_face_recognition(license, tfsdk.FACIALRECOGNITIONMODEL.TFV6, gpu_options, batch_size, 40 * mult_factor)
# benchmark_mask_detection(license, gpu_options, batch_size, 40 * mult_factor)
