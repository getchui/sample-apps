# Trueface SDK - C++ API Sample Apps
## SDK Download and Documentation
[C++ API Documentation](https://reference.trueface.ai/cpp/dev/latest/index.html)

## Getting Started
* Place Trueface include files and `libtf.a` or `libtf.so` in `./trueface_sdk/`
* Some sample apps may require you to have additional model files downloaded (they will throw an exception if the model file is not detected).
The model files can be downloaded by running the scripts in the `download_models/` directory which comes packaged as part of the SDK download bundle. 
  If you download the model files to a directory other than the build directory, you must specify the path to the directory using the `Trueface::ConfigurationOptions.modelsPath` configuration option. 


## Sample Apps
* [Landmark Detection - Live Streaming](./landmark_detection/live_streaming)
* [Landmark Detection - Draw and Save Landmarks](./landmark_detection/draw_landmarks_save_images)
* [Object Detection](./object_detection/)
* [Facial Recognition - 1:1 Identification](./facial_recognition/11/)
* [Facial Recognition - 1:N Identification](./facial_recognition/1N/)
* [Facial Recognition - 1:N Identification Read Write Video](./facial_recognition/1N_read_write_video/)
* [Facial Recognition - 1:N Identification with Thread Pool (CPU)](./facial_recognition/1N_threadpool_cpu/)
* [Facial Recognition - 1:1 Identification with Blur](./facial_recognition/blur/)
* [Head Orientation](./head_orientation/)
