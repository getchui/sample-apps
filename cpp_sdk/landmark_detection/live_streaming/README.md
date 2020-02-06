# Trueface SDK C++ Sample App
## Landmark Detection - Live Streaming
This sample app demonstrates how to detect facial landmarks and bounding boxes using the SDK. 
The landmarks and bounding boxes are drawn on the video obtained from the users webcam.

# Demo
![alt text](./demo_gifs/demo1.gif)

![alt text](./demo_gifs/demo2.gif)

### Prerequisites
Must have OpenCV installed with the `Video I/O` module built. 

### Build Instructions
* Place Trueface include files and `libtf.a` in `../trueface_sdk/`
* replace `<LICENSE_CODE>` with your license code in `src/main.cpp`
* `mkdir build && cd build`
* `cmake ..`
* `make`

### Notes
This sample was built with SDK version 0.2
