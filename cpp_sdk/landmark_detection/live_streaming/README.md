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
* Place Trueface header files in `../../trueface_sdk/include/` and trueface libraries in `../../trueface_sdk/lib/`
* Export your Trueface token to your environment as `TRUEFACE_TOKEN`.
  Alternatively, open the `CMakeLists.txt` file and edit this line here: `add_definitions(-DTRUEFACE_TOKEN="YOUR_TOKEN_HERE")`.
  Replace `YOUR_TOKEN_HERE` with the license token you were provided with. If you have not yet received a token, contact sales@trueface.ai.
* `mkdir build && cd build`
* `cmake ..`
* `make`
