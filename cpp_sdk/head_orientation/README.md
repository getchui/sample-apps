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
* Export your Trueface token to your environment as `TRUEFACE_TOKEN`.
  Alternatively, open the `CMakeLists.txt` file and edit this line here: `add_definitions(-DTRUEFACE_TOKEN="YOUR_TOKEN_HERE")`.
  Replace `YOUR_TOKEN_HERE` with the license token you were provided with. If you have not yet received a token, contact support@pangiam.com.
* `mkdir build && cd build`
* `cmake ..`
* `make`
