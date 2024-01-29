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
import face_template_quality_estimation
from observation import (Observation, ObservationCSVWriter)
from typing import List
from utils import Parameters


import tfsdk


def main() -> None:
    NUM_WARMUP = 10
    DO_WARMUP = True

    gpu_options = tfsdk.GPUOptions()
    gpu_options.enable_GPU = False  # TODO: Set this to true to benchmark on GPU.
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
    gpu_options.face_template_quality_estimator_GPU_options = gpu_module_options

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

    observations: List[Observation] = []
    parameters = Parameters(DO_WARMUP, NUM_WARMUP, 1, 200)
    preprocess_image.benchmark(gpu_options, parameters, observations)
    glasses_detection.benchmark(gpu_options, parameters, observations)

    parameters.num_iterations = 50 * mult_factor
    face_image_orientation_detection.benchmark(gpu_options, parameters, observations)

    parameters.num_iterations = 100 * mult_factor
    face_landmark_detection.benchmark(gpu_options, parameters, observations)
    detailed_landmark_detection.benchmark(gpu_options, parameters, observations)
    blink_detection.benchmark(gpu_options, parameters, observations)
    mask_detection.benchmark(gpu_options, parameters, observations)
    spoof_detection.benchmark(gpu_options, parameters, observations)
    face_template_quality_estimation.benchmark(gpu_options, parameters, observations)

    parameters.num_iterations = 200 * mult_factor
    face_image_blur_detection.benchmark(gpu_options, parameters, observations)

    parameters.num_iterations = 500 * mult_factor
    head_orientation.benchmark(gpu_options, parameters, observations)

    parameters.num_iterations = 50 * mult_factor
    for object_detection_model in (tfsdk.OBJECTDETECTIONMODEL.FAST,
                                   tfsdk.OBJECTDETECTIONMODEL.ACCURATE):
        object_detection.benchmark(gpu_options, object_detection_model, parameters, observations)

    parameters.num_iterations = 200
    face_recognition.benchmark(tfsdk.FACIALRECOGNITIONMODEL.LITE_V2, gpu_options, parameters, observations)
    face_recognition.benchmark(tfsdk.FACIALRECOGNITIONMODEL.LITE_V3, gpu_options, parameters, observations)

    parameters.num_iterations = 40 * mult_factor
    face_recognition.benchmark(tfsdk.FACIALRECOGNITIONMODEL.TFV5_2, gpu_options, parameters, observations)
    face_recognition.benchmark(tfsdk.FACIALRECOGNITIONMODEL.TFV6, gpu_options, parameters, observations)
    face_recognition.benchmark(tfsdk.FACIALRECOGNITIONMODEL.TFV7, gpu_options, parameters, observations)
    # Benchmarks with batching.
    # Only run for GPU inference
    # CPU inference does not support batching, so will provide no speedup
    if gpu_options.enable_GPU:
        parameters.batch_size = batch_size
        face_recognition.benchmark(tfsdk.FACIALRECOGNITIONMODEL.TFV5_2, gpu_options, parameters, observations)
        face_recognition.benchmark(tfsdk.FACIALRECOGNITIONMODEL.TFV6, gpu_options, parameters, observations)
        face_recognition.benchmark(tfsdk.FACIALRECOGNITIONMODEL.TFV7, gpu_options, parameters, observations)

        parameters.num_iterations = 100 * mult_factor
        mask_detection.benchmark(gpu_options, parameters, observations)
        face_template_quality_estimation.benchmark(gpu_options, parameters, observations)
        detailed_landmark_detection.benchmark(gpu_options, parameters, observations)
        blink_detection.benchmark(gpu_options, parameters, observations)
        face_image_orientation_detection.benchmark(gpu_options, parameters, observations)
        face_image_blur_detection.benchmark(gpu_options, parameters, observations)


    ObservationCSVWriter('benchmarks.csv').write(observations)


if __name__ == '__main__':
    main()
