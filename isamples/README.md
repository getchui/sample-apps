# Face Recognition and Spoof Detection Project

This project demonstrates the implementation of a face recognition and spoof detection system using Swift and Trueface SDK. The app features multiple view controllers for face detection, spoof detection, face recognition, enrolling new people, and managing people's identities.

## Table of Contents
* Getting Started
* Prerequisites
* Installation
* Usage
* Contributing
* License

## Getting Started
These instructions will help you set up the project on your local machine for development and testing purposes.

## Prerequisites
* Xcode 13 or later
* iOS 15.0 or later
* Swift 5 or later
* A physical iOS device (some functionalities may not work on the simulator)

## Installation
1. Clone the repository to your local machine:

git clone https://github.com/getchui/sample-apps.git

2. Open the project in Xcode:

cd isamples
open isamples.xcodeproj

3. Ensure that your iOS device is connected to your computer, and the device is selected as the build target in Xcode.
4. Download the Trueface SDK for iOS from the official website, and follow the instructions to install it in your project.
5. Obtain a license for the Trueface SDK, and set it up in your project according to the SDK's documentation.
6. Download the following models and add them to your project: 
    - blink_detector_v3
    - face_landmark_detector_v2
    - face_recognition_lite_v2
    - face_detector_v2_fast
    - mask_detector_v3
    - spoof_v6


## Usage
Build and run the app on your iOS device.

The app will open with multiple options for face detection, spoof detection, face recognition, enrolling new faces, and managing people's identities.

Choose the desired option, and follow the on-screen instructions to perform the selected operation.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
