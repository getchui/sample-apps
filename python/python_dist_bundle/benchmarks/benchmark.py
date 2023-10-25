# Run benchmarks

import preprocess_image
import face_image_orientation_detection
import face_landmark_detection
import detailed_landmark_detection
import head_orientation
import face_image_blur_detection
import blink_detection
import mask_detection
import glasses_detection
import spoof_detection
import object_detection
import face_recognition
from utils import (Parameters, Stopwatch)

import tfsdk

import os

# ********************************************************************************************************
# ********************************************************************************************************
#
#                                  START OF BENCHMARKING SCRIPT
#
# ********************************************************************************************************
# ********************************************************************************************************

def main():
    NUM_WARMUP = 10
    DO_WARMUP = True

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

    parameters = Parameters(DO_WARMUP, NUM_WARMUP, 1, 200)
    preprocess_image.benchmark(license, gpu_options, parameters)
    glasses_detection.benchmark(license, gpu_options, parameters)

    parameters.num_iterations = 50 * mult_factor
    face_image_orientation_detection.benchmark(license, gpu_options, parameters)

    parameters.num_iterations = 100*mult_factor
    face_landmark_detection.benchmark(license, gpu_options, parameters)
    detailed_landmark_detection.benchmark(license, gpu_options, parameters)
    blink_detection.benchmark(license, gpu_options, parameters)
    mask_detection.benchmark(license, gpu_options, parameters)
    spoof_detection.benchmark(license, gpu_options, parameters)

    parameters.num_iterations = 200*mult_factor
    face_image_blur_detection.benchmark(license, gpu_options, parameters)

    parameters.num_iterations = 500*mult_factor
    head_orientation.benchmark(license, gpu_options, parameters)

    parameters.num_iterations = 100*mult_factor
    for object_detection_model in (tfsdk.OBJECTDETECTIONMODEL.FAST,
                                   tfsdk.OBJECTDETECTIONMODEL.ACCURATE):
        object_detection.benchmark(license, gpu_options, object_detection_model, parameters)

    parameters.num_iterations = 200
    if gpu_options.enable_GPU is False:
        # get_face_feature_vectors is not supported by the LITE model
        face_recognition.benchmark(license, tfsdk.FACIALRECOGNITIONMODEL.LITE, gpu_options, parameters)

    face_recognition.benchmark(license, tfsdk.FACIALRECOGNITIONMODEL.LITE_V2, gpu_options, parameters)

    parameters.num_iterations = 40*mult_factor
    face_recognition.benchmark(license, tfsdk.FACIALRECOGNITIONMODEL.TFV5_2, gpu_options, parameters)
    face_recognition.benchmark(license, tfsdk.FACIALRECOGNITIONMODEL.TFV6, gpu_options, parameters)
    face_recognition.benchmark(license, tfsdk.FACIALRECOGNITIONMODEL.TFV7, gpu_options, parameters)
    # Benchmarks with batching.
    # On CPU, should be the same speed as a batch size of 1.
    # On GPU, will increase the throughput.
    parameters.batch_size = batch_size
    face_recognition.benchmark(license, tfsdk.FACIALRECOGNITIONMODEL.TFV5_2, gpu_options, parameters)
    face_recognition.benchmark(license, tfsdk.FACIALRECOGNITIONMODEL.TFV6, gpu_options, parameters)
    face_recognition.benchmark(license, tfsdk.FACIALRECOGNITIONMODEL.TFV7, gpu_options, parameters)

    parameters.num_iterations = 100*mult_factor
    mask_detection.benchmark(license, gpu_options, parameters)


if __name__ == '__main__':
    main()
