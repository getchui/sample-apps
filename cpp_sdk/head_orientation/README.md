# Trueface SDK C++ Sample App
## Head Orientation
This sample app demonstrates how to use the SDK to estimate the head orientation.
The yaw, pitch, and roll are first obtained. Next, axis are draw on the video obtained from the users webcam. 

# Demo
![alt text](./demo_gifs/demo1.gif)

### Prerequisites
Must have OpenCV installed with the `Video I/O` module built. 

### Build Instructions
* Place Trueface header files in `../trueface_sdk/include/` and trueface libraries in `../trueface_sdk/lib/`
* replace `<LICENSE_CODE>` with your license code in `src/main.cpp`
* `mkdir build && cd build`
* `cmake ..`
* `make`
