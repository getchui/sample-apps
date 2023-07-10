# Trueface SDK C++ Sample App
## Facial Recognition - 1:1 Identification
This sample app demonstrates how to generate templates from images loaded from file.
The templates are then compared to generate a similarity score.

### Demo
![alt text](https://i.ibb.co/G2skdHJ/Untitled-presentation-1.jpg)

![alt text](https://i.ibb.co/SPwVK4V/Untitled-presentation-1.jpg)

### Build Instructions
* Place Trueface header files in `../../trueface_sdk/include/` and trueface libraries in `../../trueface_sdk/lib/`
* Download the appropriate face recognition model using the download scripts which come bundled with the SDK. 
* Export your Trueface token to your environment as `TRUEFACE_TOKEN`.
  Alternatively, open the `CMakeLists.txt` file and edit this line here: `add_definitions(-DTRUEFACE_TOKEN="YOUR_TOKEN_HERE")`.
  Replace `YOUR_TOKEN_HERE` with the license token you were provided with. If you have not yet received a token, contact support@pangiam.com.
* `mkdir build && cd build`
* `cmake ..`
* `make`

