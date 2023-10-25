from stopwatch import Stopwatch
import tfsdk

import os

NUM_WARMUP = 10
DO_WARMUP = True

def benchmark(license, gpu_options, num_iterations = 200):
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

    # Now repeat with already decoded images (ex. you grab an image from your video stream).
    if DO_WARMUP:
        for _ in range(NUM_WARMUP):
            error_code, img = sdk.preprocess_image(img.get_data(), img.get_width(), img.get_height(), tfsdk.COLORCODE.rgb)
            if error_code != tfsdk.ERRORCODE.NO_ERROR:
                print('Error: Unable to preprocess image')
                return

    stop_watch = Stopwatch()
    for _ in range(num_iterations):
        sdk.preprocess_image(img.get_data(), img.get_width(), img.get_height(), tfsdk.COLORCODE.rgb)
    total_time = stop_watch.elapsedTimeMilliSeconds()
    avg_time = total_time / num_iterations

    print("Average time preprocessImage RGB pixel array in memory ({}x{}): {} ms | {} iterations".format(
        img.get_width(), img.get_height(), avg_time, num_iterations))