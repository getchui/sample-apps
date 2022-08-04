# Run benchmarks 

import tfsdk
import os
from colorama import Fore
from colorama import Style
from time import time
import inspect

import time

def current_milli_time():
    return round(time.time() * 1000)

def benchmark_preprocess_image(license, gpu_options, num_iterations = 200):
    options = tfsdk.ConfigurationOptions()
    options.models_path = os.getenv('MODELS_PATH') or './'
    options.GPU_options = gpu_options

    sdk = tfsdk.SDK(options)

    is_valid = sdk.set_license(os.environ['TRUEFACE_TOKEN'])
    if (is_valid == False):
        print(f"{Fore.RED}Invalid License Provided{Style.RESET_ALL}")
        print(f"{Fore.RED}Be sure to export your license token as TRUEFACE_TOKEN{Style.RESET_ALL}")
        quit()


    ret, img = sdk.preprocess_image("./headshot.jpg")
    if (ret != tfsdk.ERRORCODE.NO_ERROR):
        print("There was an error setting the image in the benchmark_preprocess_image method")
        quit()

    # Run our timing code
    t1 = current_milli_time()
    for i in range(num_iterations):
        ret, img = sdk.preprocess_image("./headshot.jpg")
    t2 = current_milli_time()

    total_time = t2 - t1
    avg_time = total_time / num_iterations

    print("Average time to preprocess image ({}x{}): {} ms | {} iterations".format(img.get_width(), img.get_height(), avg_time, num_iterations))
    
def benchmark_face_landmark_detection(license, gpu_options, num_iterations = 100):
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

    # Run our timing code
    t1 = current_milli_time()
    for i in range(num_iterations):
        faces = sdk.detect_faces(img)
    t2 = current_milli_time()

    total_time = t2 - t1
    avg_time = total_time / num_iterations

    print("Average time face and landmark detection ({}x{}): {} ms | {} iterations".format(img.get_width(), img.get_height(), avg_time, num_iterations))
    
def benchmark_detailed_landmark_detection(license, gpu_options, num_iterations = 100):
    options = tfsdk.ConfigurationOptions()
    options.models_path = os.getenv('MODELS_PATH') or './'
    options.GPU_options = gpu_options

    options.smallest_face_height = 40
    options.initialize_module.face_detector = True
    options.initialize_module.landmark_detector = True

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
        ret, landmarks = sdk.get_face_landmarks(img, fb)
    t2 = current_milli_time()

    if ret != tfsdk.ERRORCODE.NO_ERROR:
        print("Unable to run 106 face landmark detection in {} method".format(inspect.stack()[0][3]))
        quit()

    total_time = t2 - t1
    avg_time = total_time / num_iterations

    print("Average time 106 face landmark detection: {} ms | {} iterations".format(avg_time, num_iterations))

def benchmark_blink_detection(license, gpu_options, num_iterations = 100):
    options = tfsdk.ConfigurationOptions()
    options.models_path = os.getenv('MODELS_PATH') or './'
    options.GPU_options = gpu_options

    options.smallest_face_height = 40
    options.initialize_module.face_detector = True
    options.initialize_module.liveness = True

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
        ret, mask_labels = sdk.detect_masks(chips)
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

    # Run our timing code
    t1 = current_milli_time()
    for i in range(num_iterations):
        ret, yaw, pitch, roll = sdk.estimate_head_orientation(img, fb)
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

    print("Average time face recognition: {} ms | batch size = {} | {} iterations".format(fr_model.name, avg_time, batch_size, num_iterations))

# ********************************************************************************************************
# ********************************************************************************************************
#
#                                  START OF BENCHMARKING SCRIPT
#
# ********************************************************************************************************
# ********************************************************************************************************

license = os.environ['TRUEFACE_TOKEN']

gpu_options = tfsdk.GPUOptions()
gpu_options.enable_GPU = False # TODO: Set this to true to benchmark on GPU. 
gpu_options.device_index = 0

gpu_module_options = tfsdk.GPUModuleOptions()
gpu_module_options.precision = tfsdk.PRECISION.FP16

batch_size = 4
gpu_module_options.max_batch_size = batch_size
gpu_module_options.opt_batch_size = batch_size

gpu_options.face_detector_GPU_options = gpu_module_options
gpu_options.face_recognizer_GPU_options = gpu_module_options
gpu_options.mask_detector_GPU_options = gpu_module_options

mult_factor = 1

if (gpu_options.enable_GPU):
    print("Using GPU for inference")
    mult_factor = 10
else:
    print("Using CPU for inference")

benchmark_preprocess_image(license, gpu_options)
benchmark_face_landmark_detection(license, gpu_options)
benchmark_detailed_landmark_detection(license, gpu_options)
benchmark_blink_detection(license, gpu_options)
benchmark_spoof_detection(license, gpu_options)
benchmark_mask_detection(license, gpu_options, 1, 100 * mult_factor)
benchmark_head_orientation(license, gpu_options)
benchmark_object_detection(license, gpu_options)

if gpu_options.enable_GPU == False:
    # get_face_feature_vectors method is not support by the LITE and LITE_V2 models
    benchmark_face_recognition(license, tfsdk.FACIALRECOGNITIONMODEL.LITE, gpu_options, 1, 200)
    benchmark_face_recognition(license, tfsdk.FACIALRECOGNITIONMODEL.LITE_V2, gpu_options, 1, 200)

benchmark_face_recognition(license, tfsdk.FACIALRECOGNITIONMODEL.TFV6, gpu_options, 1, 40 * mult_factor)
benchmark_face_recognition(license, tfsdk.FACIALRECOGNITIONMODEL.TFV5, gpu_options, 1, 40 * mult_factor)
benchmark_face_recognition(license, tfsdk.FACIALRECOGNITIONMODEL.FULL, gpu_options, 1, 40 * mult_factor)

# TODO Cyrus: Add Object detection benchmarking coe


# Benchmarks with batching.
# On CPU, should be the same speed as a batch size of 1.
# On GPU, will increase the throughput.
benchmark_face_recognition(license, tfsdk.FACIALRECOGNITIONMODEL.TFV6, gpu_options, batch_size, 40 * mult_factor)
benchmark_mask_detection(license, gpu_options, batch_size, 40 * mult_factor)
