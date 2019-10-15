# Trueface SDK C++ Sample App
## Facial Recognition - Blur
This sample app demonstrates the use of 1:N matching with blur. Templates are first generated and enrolled into a gallery with their associated ID.
Next, all faces are extracted from the camera frame and used to generate templates. If any of these templates match those in the gallery, the identity is draw on the image.
If an identify is not recognized, the face is blurred 

### Demo
![alt text]()

![alt text]()

### Prerequisites
Must have OpenCV installed with the `Video I/O` module built. 

### Build Instructions
* Place Trueface include files and `libtf.a` in `../../trueface_sdk/`
* replace `<LICENSE_CODE>` with your license code in `src/main.cpp`
* `mkdir build && cd build`
* `cmake ..`
* `make`

### Notes
This sample was built with SDK version 0.2