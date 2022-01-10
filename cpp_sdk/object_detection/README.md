# Trueface SDK C++ Sample App
## Object Detection
This sample app demonstrates how to detect objects using the SDK. The object labels are drawn on the video stream.

# Demo
![alt text](./demo_gifs/demo1.gif)

![alt text](./demo_gifs/demo2.gif)

### Prerequisites
Must have OpenCV installed with the `Video I/O` module built. 

### Build Instructions
* Place Trueface header files in `../trueface_sdk/include/` and trueface libraries in `../trueface_sdk/lib/`
* replace `<LICENSE_CODE>` with your license code in `src/main.cpp`
* `mkdir build && cd build`
* `cmake ..`
* `make`
