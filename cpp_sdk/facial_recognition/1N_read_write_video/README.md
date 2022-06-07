# Trueface SDK C++ Sample App
## Facial Recognition - 1:N Read Write Video
This sample app demonstrates the use of 1:N matching. 
Face Recognition templates are first generated and enrolled into a collection with their associated ID.
The quality of these enrollment images are checked before enrolling into the collection as we want to ensure only high quality images are enrolled in collections.
The video is then read from disk, and face recognition is then run on each frame of the video.
If any of the identities match those in the collection, the identity and bounding box is draw on the video stream.
The video stream is then displayed and written to disk.

### Demo
![alt text](./demo_gifs/demo1.gif)

![alt text](./demo_gifs/demo2.gif)

![alt text](./demo_gifs/demo3.gif)

![alt text](./demo_gifs/demo4.gif)

### Prerequisites
Must have OpenCV installed with the `Video I/O` module built. 

### Build Instructions
* Place Trueface header files in `../../trueface_sdk/include/` and trueface libraries in `../../trueface_sdk/lib/`
* Download the appropriate face recognition model using the download scripts which come bundled with the SDK. 
* Export your Trueface token to your environment as `TRUEFACE_TOKEN`.
  Alternatively, open the `CMakeLists.txt` file and edit this line here: `add_definitions(-DTRUEFACE_TOKEN="YOUR_TOKEN_HERE")`.
  Replace `YOUR_TOKEN_HERE` with the license token you were provided with. If you have not yet received a token, contact sales@trueface.ai.
* `mkdir build && cd build`
* `cmake ..`
* `make`
