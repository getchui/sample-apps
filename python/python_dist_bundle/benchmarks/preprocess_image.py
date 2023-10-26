from utils import (Parameters, Stopwatch, SDKFactory)
import tfsdk

import os


def benchmark(gpu_options: tfsdk.GPUOptions, parameters: Parameters) -> None:
    # Initialize the SDK
    sdk = SDKFactory.createSDK(gpu_options, initialize_modules=[])

    img_path = './headshot.jpg'

    # First run the benchmark for an image on disk
    # Run once to ensure everything works
    if parameters.do_warmup:
        for _ in range(parameters.num_warmup):
            error_code, img = sdk.preprocess_image(img_path)
            if error_code != tfsdk.ERRORCODE.NO_ERROR:
                print("Unable to preprocess the image")
                return

    # Time the preprocess_image function
    stop_watch = Stopwatch()
    for _ in range(parameters.num_iterations):
        sdk.preprocess_image(img_path)
    total_time = stop_watch.elapsedTimeMilliSeconds()
    avg_time = total_time / parameters.num_iterations

    print("Average time preprocess_image JPG image from disk ({}x{}): {} ms | {} iterations".format(
        img.get_width(), img.get_height(), avg_time, parameters.num_iterations))

    # Now repeat with encoded image in memory
    size = os.path.getsize(img_path)
    with open(img_path, 'rb') as infile:
        data = infile.read(size)

    buffer = []
    for b in data:
        buffer.append(b)

    if parameters.do_warmup:
        for _ in range(parameters.num_warmup):
            error_code, img = sdk.preprocess_image(buffer)
            if error_code != tfsdk.ERRORCODE.NO_ERROR:
                print("Unable to preprocess the image")
                return

    # Time the preprocess_image function
    stop_watch = Stopwatch()
    for _ in range(parameters.num_iterations):
        sdk.preprocess_image(buffer)
    total_time = stop_watch.elapsedTimeMilliSeconds()
    avg_time = total_time / parameters.num_iterations

    print("Average time preprocess_image JPG image in memory ({}x{}): {} ms | {} iterations".format(
        img.get_width(), img.get_height(), avg_time, parameters.num_iterations))

    # Now repeat with already decoded images (ex. you grab an image from your video stream).
    if parameters.do_warmup:
        for _ in range(parameters.num_warmup):
            error_code, img = sdk.preprocess_image(img.get_data(), img.get_width(), img.get_height(), tfsdk.COLORCODE.rgb)
            if error_code != tfsdk.ERRORCODE.NO_ERROR:
                print('Error: Unable to preprocess image')
                return

    stop_watch = Stopwatch()
    for _ in range(parameters.num_iterations):
        sdk.preprocess_image(img.get_data(), img.get_width(), img.get_height(), tfsdk.COLORCODE.rgb)
    total_time = stop_watch.elapsedTimeMilliSeconds()
    avg_time = total_time / parameters.num_iterations

    print("Average time preprocessImage RGB pixel array in memory ({}x{}): {} ms | {} iterations".format(
        img.get_width(), img.get_height(), avg_time, parameters.num_iterations))
