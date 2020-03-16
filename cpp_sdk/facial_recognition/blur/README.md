# Trueface SDK C++ Sample App
## Facial Recognition - 1:N Identification with Blur
This sample app demonstrates the use of 1:N identification with blur. Templates are first generated and enrolled into a collection with their associated ID.
Next, all faces are extracted from the camera frame and used to generate templates. If any of these templates match those in the collection, the identity is draw on the image.
If an identify is not recognized, the face is blurred 

### Demo
![alt text](./demo_gifs/demo1.gif)

![alt text](./demo_gifs/demo2.gif)

![alt text](./demo_gifs/demo3.gif)

### Prerequisites
Must have OpenCV installed with the `Video I/O` module built. 

### Build Instructions
* Place Trueface include files and `libtf.a` in `../../trueface_sdk/`
* replace `<LICENSE_CODE>` with your license code in `src/main.cpp`
* `mkdir build && cd build`
* `cmake ..`
* `make`
