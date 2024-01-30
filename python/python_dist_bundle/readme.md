# Python Bindings Sample Code

## Prerequisites

In order to run the sample apps, you must place the python bindings library in the same directory as the python script. 
Alternatively, you can add the absolute path to the directory containing the python bindings library to your `PYTHONPATH` environment variable.
You must also add the aboslute path to the directory containing the provided ONNX Runtime dependency library to your `LD_LIBRARY_PATH` environment variable if using Linux or `DYLD_FALLBACK_LIBRARY_PATH` if using MacOS.

You can add to an environment variable as follows: `export PYTHONPATH=$PYTHONPATH:/path/to/directory/containing/tfsdk...`.

The following dependencies may need to be installed for some sample apps:
- `pip install numpy`
- `pip install colorama`
- `pip install opencv-python`
- Optional: `pip install cupy` (required for `face_recognition_vram.py` sample app)

If you are are using the GPU SDK, be sure to read [this guide](https://reference.trueface.ai/cpp/dev/latest/index.html#gpu-sdk-dependencies) on installing GPU dependencies.

## Benchmarks

The benchmarking script can be found in the ./benchmarks directory.
To benchmark GPU inference, change [this](https://github.com/getchui/sample-apps/blob/a14337d5bcbdfcee952293da11c62ae7be239504/python/python_dist_bundle/benchmarks/benchmark.py#L387) line to `True`. 

## Running The Sample Code

Start by exporting your provided token as an envionrmnet variable with the key `TRUEFACE_TOKEN`. 
ex. `export TRUEFACE_TOKEN=<YOUR_TOKEN_HERE>`

Most Some sample apps may require you to have additional model files downloaded (they will throw an exception if the model file is not detected).
The model files can be downloaded by running `./download_models/download_all_models.sh`. 
You can also choose to download only the models files required for your application by running the individual scripts provided in that directory.
If you download the model files to a directory other than the run directory, you must specify the path to the directory using the `tfsdk.ConfigurationOptions.models_path` configuration option.

The sample code can be run by calling `python3 sample_code_name.py`.

If the python interpreter is unable to find `tfsdk`, ensure the version of the library
you are using matches your python version (including minor version) exactly.
Additionally, make sure that the path to the directory containing the python bindings library is in your `PYTHONPATH`.

The sample code located in sub-directories should be run from the current directory. Ex. `python3 identification_1_n/identification_1_n.py`.

# Sample Code

**Note**
Any sample app ending in `_live_streaming.py` requires you to have a camera (physical or RTSP).

### Chip Extraction
- `chip_extraction.py` demonstrates how to extract and display an aligned 112x112 chip from a face image.

### Facial Landmarks
- `face_landmarks_live_streaming.py` uses OpenCV to grab frames from the webcam, runs face detection on the frame, then gets the 106 facial landmarks for each of the detected faces. Finally, the facial landmarks are drawn on the video frame which is then displayed.

### Active Spoof Detection
- `active_spoof.py` demonstrates how to run active spoof. 
  Active spoof works by analyzing the way a persons face changes as they move closer to a camera. 
  The active spoof solution therefore required two images and expects the face a certain distance from the camera. 
  In the far image, the face should be about 18 inches from the camera, while in the near image, the face should be 7-8 inches from the camera.
  You can find a more involved sample app [here](https://github.com/getchui/sample-apps/tree/master/python/active_spoof_frontend_app).

### Spoof Detection
- `spoof_detection_live_streaming.py` demonstrates how to run spoof detection on a video feed. 
  Spoof is detected when a user is trying to impersonate an identity, for example by holding a phone with a face image up to the camera. The input image must meet strict criteria in order for it to pass the spoof test. 

### Face Recognition - 1 to 1 Identification
- `face_recognition.py` demonstrates the usage of our face recognition API for 1 to 1 matching. 
   A convenient method is provided that loads images from disk, `set_image()`. This method can decode JPEG, PNG, BMP and PPM images. 

- `face_recognition_image_buffer.py` uses the overloaded `set_image()` method that takes the numpy array to a decoded image buffer in memory.
  The main usecase is processing frames grabbed from a camera without writing them into a file.

### Face Recognition - 1 to N Identification
- `identification_1_n/enroll_in_database.py` demonstrates how you can generate face recognition templates and then enroll them into a SQLite database on disk.
  You can easily adapt this sample app to enroll templates into a PostgreSQL database.
  
- `identification_1_n/identification_1_n.py` demonstrates how you can load a database of face recognition templates and then run a search query against that database.
  
- `identification_1_n/identification_1_n_live_streaming.py` is the same as `identification_1_n/identification_1_n.py`, but uses OpenCV to grab frames from the webcam and runs search queries using those images. 
The annotated video is then displayed in real time. 

- `identification_1_n/identification_multiple_collections.py` demonstrates how to load multiple collections into memory and how to perform operations (enrollment, identification) on those collections.

Note, `enroll_in_database.py` must be run before you can run `identification_1_n.py` or `identification_1_n_live_streaming.py`.

### Object Detection
- `object_detection.py` demonstrates how to use the object detection module on static images.
  
- `object_detection_live_streaming.py` is the same as `object_detection.py`, but uses OpenCV to grab frames from the webcam and runs object detection on these images.
  The annotated video is then displayed in real time.

### Blur
- `blur_live_streaming.py` uses OpenCV to grab frames from the webcam then runs face detection. The detected faces are then blurred.
The annotated video is then displayed in real time.

### Head Pose Estimation
- `head_pose_estimation_live_stream.py` uses OpenCV to grab frames from the webcam then draws the head pose axes on the live stream. 

### Liveness
- `blink_detection_live_streaming.py` demonstrates the use of our blink detection API.
Passing multiple frames in succession to the blink detector can be used to build a liveness check.

### Mask Detection
- `detect_mask.py` demonstrates how to run mask detection on an image. Mask is detected when a user is wearing a face mask.

### Eye glasses Detection
- `detect_eyeglasses.py` demonstrates how to run eye glasses detection on an image. Glasses are detected when a user is wearing some type of glasses.

### GPU Sample Code - Batch Inference and VRam
- `gpu_sample_apps/face_recognition_batching.py` demonstrates how to generate face recognition templates and run mask detection in batch when using the GPU library.
Batching is used to increase throughput on GPUs.

- `gpu_sample_apps/face_recognition_vram.py` demonstrates how the GPU SDK can be used to detect the largest face in an image already loaded in the graphics card's memory.
  NOTE: You should only reference this sample code if your image is already in GPU memory, for example, decoding video directly to GPU memory.
  In order to run this sample code, you must have `cupy` installed.

- `batch_enroll_in_database.py` demonstrates how to generate feature vectors in batch and enroll them into a collection. This example is perfect for processing large numbers of images offline.

## Running Sample Code from Github Repo
This sample code comes shipped as part of the SDK download bundle. 
If you are instead running this sample code by cloning the Trueface sample apps github repo [here](https://github.com/getchui/sample-apps), you will need to make a few modifications to get things working:

* Download the python bindings library and supporting Trueface libraries and place them in this directory.
* Obtain the scripts for downloading the model files from the SDK download bundle.
