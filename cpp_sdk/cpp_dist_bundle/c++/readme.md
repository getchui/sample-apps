# C++ SDK Sample Code

## Prerequisites
You must have CMake 3.15 installed on your system as well as a C++ compiler (ex. gcc, clang, MSVC).

If you are are using the GPU SDK, be sure to read [this guide](https://reference.trueface.ai/cpp/dev/latest/index.html#gpu-sdk-dependencies) on installing GPU dependencies. 


## Build Instructions
Export your Trueface token to your environment as `TRUEFACE_TOKEN`. 
Be sure to wrap the token in quotes `"`. 
So for example, `export TRUEFACE_TOKEN="YOUR_TOKEN_HERE"`. 
Alternatively, open the `CMakeLists.txt` file and edit this line here: `add_definitions(-DTRUEFACE_TOKEN="YOUR_TOKEN_HERE")`.
Replace `YOUR_TOKEN_HERE` with the license token you were provided with. If you have not yet received a token, contact support@pangiam.com. 
This will insert your token into all the sample apps. Alternatively, you can edit each individual sample app and enter your token manually. 

### Linux
Run the following commands to compile and link the sample apps:
- `mkdir build`
- `cd build`
- `cmake ..`
- `make -j $(nproc)`

Before you can run the sample apps, you must add the aboslute path to the directory containing ONNX Runtime (`../../trueface_sdk/lib/`) to your `LD_LIBRARY_PATH` environment variable. So for example:
- `cd ../../../trueface_sdk/lib/`
- `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)`
- `cd -`

### MacOS
Run the following commands to compile and link the sample apps:
- `mkdir build`
- `cd build`
- `cmake ..`
- `make -j $(nproc)`

Before you can run the sample apps, you must add the aboslute path to the directory containing ONNX Runtime (`../../trueface_sdk/lib/`) to your `DYLD_FALLBACK_LIBRARY_PATH` environment variable. So for example:
- `cd ../../../trueface_sdk/lib/`
- `export DYLD_FALLBACK_LIBRARY_PATH=$DYLD_FALLBACK_LIBRARY_PATH:$(pwd)`
- `cd -`

### Windows
If building on Windows, be sure to first read our guide [here](https://reference.trueface.ai/cpp/dev/latest/index.html#windows-sdk). Next, you will need to run the following commands:
- `mkdir build`
- `cd build`
- `cmake -A x64 ..`
- `cmake --build . --config release --parallel 4`
- After building the executables, you will need to copy them from `build/Release` to `build/` before running them.
- Additionally, you must copy over all `.dll` libs located in `../trueface_sdk/lib` to the build directory. 
- A peculiarity about Window's is that when an exception is thrown, no error message is printed to the console. You may therefore choose to wrap the sample code in try catch statements, printing the exception messages. As an example, if the appropriate model file is not found, the SDK will throw an exception.

## Running the Sample Apps
At this point, you are ready to run the sample apps. 
Most sample apps require you to have additional model files downloaded (they will throw an exception if the modeOl file is not detected).
All the model files can be downloaded by running `../../download_models/download_all_models.sh`. You can also choose to download only the models files required for your application by running the individual scripts provided in that directory. If you download the model files to a directory other than the build directory, you must specify the path to the directory using the `Trueface::ConfigurationOptions.modelsPath` configuration option. 

## Note
- When compiling your application, be sure to use the `-Ofast` compiler flag. 

## Other CMake Options
- `BUILD_GPU`: Builds sample apps involving GPU specific functions. Will need the GPU SDK in order to run these sample apps. ex. `cmake -DBUILD_GPU=ON ..`
- `BUILD_GPU_OPENCV`: Builds the sample app which demonstrates loading images directly from GPU VRAM. ex. `cmake -DBUILD_GPU_OPENCV=ON`. 
  Enabling this option will also enable `BUILD_GPU`. 
  In order to enable this option, you must have OpenCV with CUDA installed on your system. 
  You can build and install this by navigating to the `3rd_party_libs` directory and running `build_opencv.sh` which will build and install OpenCV CUDA with the required Contrib modules.

## Cross Compiling
We currently support cross compiling for `aarch64-linux-gnu` and `arm-linux-gnueabihf`. Start by downloading the appropriate SDK for the target architecture
Next, invoke the cmake command by specify the appropriate toolchain file:
- aarch64: `cmake -DCMAKE_TOOLCHAIN_FILE=../tools/aarch64-linux-gnu.toolchain.cmake ..`
- aarch32: `cmake -DCMAKE_TOOLCHAIN_FILE=../tools/arm-linux-gnueabihf.toolchain.cmake ..`

## Sample Code
### Face Recognition - 1 to 1 Identification
- `face_recognition_1_1.cpp` demonstrates the usage of our face recognition API for 1 to 1 matching. A convenient method is provided that loads images from disk, `setImage()`. This method can decode JPEG, PNG, BMP and PPM images.
- `face_recognition_1_1_image_buffer.cpp` uses the overloaded `setImage()` method that takes the pointer to a decoded image buffer in memory. 
The main usecase is processing frames grabbed from a camera without writing them into a file. 

### Face Recognition - 1 to N Identification

The 1 to N identification sample app consists of two parts: 
- `identification_1_n/enroll_in_database.cpp` demonstrates how you can generate face recognition templates and then enroll them into a SQLite database on disk.
   You can easily adapt this sample app to enroll templates into a PostgreSQL database.
- `identification_1_n/identification_1_n.cpp` demonstrates how you can load a database of face recognition templates and then run a search query against that database.
  
Note, `enroll_in_database` must be run before you can run `identification_1_n`.

- `identification_1_n/multiple_collections.cpp` demonstrates how to load multiple collections into memory at once, and how to perform operations (enrollment, identification) on said collections. 

### Object Detection 
- `object_detection.cpp` demonstrates how to run object detection on an image, then prints the label for all the detected objects.

### Liveness
- `blink_detection.cpp` demonstrates the use of our blink detection API. 
Passing multiple frames in succession to the blink detector can be used to build a liveness check. 

### Active Spoof Detection
- `active_spoof.cpp` demonstrates how to run active spoof. 
  Active spoof works by analyzing the way a persons face changes as they move closer to a camera. 
  The active spoof solution therefore required two images and expects the face a certain distance from the camera. 
  In the far image, the face should be about 18 inches from the camera, while in the near image, the face should be 7-8 inches from the camera.

### Spoof Detection
- `spoof_detection.cpp` demonstrates how to identify presentation attacks, such as someone holding their phone or a physical picture up to the camera. 

### Mask Detection
- `detect_mask.cpp` demonstrates how to run mask detection on an image. Mask is detected when a user is wearing a face mask.

### Eye glasses Detection
- `detect_eyeglasses.cpp` demonstrates how to run eye glasses detection on an image. Glasses are detected when a user is wearing some type of glasses.

### GPU Sample Code - Batch Inference and VRam
- `gpu_sample_apps/batch_fr_cuda.cpp` demonstrates how to generate face recognition templates and run mask detection in batch when using the GPU library.
Batching is used to increase throughput on GPUs. 

- `gpu_sample_apps/face_recognition_image_in_vram.cpp` demonstrates how the GPU SDK can be used to run face detection and face recognition with an image already loaded in the graphics card's memory.
  In order to run this sample code, you must have OpenCV with CUDA installed on your system.
  You can build and install this by navigating to the `3rd_party_libs` directory and running `build_opencv.sh` which will build and install OpenCV CUDA with the required Contrib modules.


## Live Streaming and Video Decoding
There are many live-streaming examples on our [Sample Apps Page](https://github.com/getchui/sample-apps) under `/cpp_sdk`.
These require you to have OpenCV installed on your system. 
