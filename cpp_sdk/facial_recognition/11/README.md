# Trueface SDK C++ Sample App
## Facial Recognition - 1:1 Identification
This sample app demonstrates how to generate templates from images loaded from file.
The templates are then compared to generate a similarity score.

### Demo
![alt text](https://i.ibb.co/G2skdHJ/Untitled-presentation-1.jpg)

![alt text](https://i.ibb.co/SPwVK4V/Untitled-presentation-1.jpg)

### Build Instructions
* Place Trueface include files and `libtf.a` in `../../trueface_sdk/`
* Download the appropriate face recognition model using the download scripts which come bundled with the SDK. 
* replace `<LICENSE_CODE>` with your license code in `src/main.cpp`
* `mkdir build && cd build`
* `cmake ..`
* `make`

