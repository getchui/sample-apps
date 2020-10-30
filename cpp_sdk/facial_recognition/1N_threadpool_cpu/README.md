# Trueface SDK C++ Sample App
## Facial Recognition - 1:N Identification with Thread pool (CPU)
This sample apps demonstrates how to build a scalable architecture for 1 to N search using a thread pool with CPU only.
For the sake of demonstration, the entire app is contained within a single process (hence the use of a thread pool).
However, for a more scalable approach, separate each worker thread into it's own microservice process, and use a Message Queue system
such as Redis to facilitate communication between the processes. This allows for the microservices to be distributed across multiple devices.

This sample app is ideal for beefy CPUs with many cores (it was tested with Dual Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz - 20 cores total - 40 threads total).
It is not advised to create more than once instance of the SDK for parallel CPU inference on CPUs with 4 or less cores.

Each SDK instance can use up to 8 threads for inference, so create the right number of SDK instances for your CPU.

This sample app will NOT show how to enroll new face templates into a collection. It will assume that you already have a sqlite / postgres 
database which has a collection full of identities. We will therefore be loading the collection and assuming it is populated.
Refer to the other 1 to N sample apps to learn how to enroll templates.
It will instead show how to consume and process multiple video streams to search for identities in those video streams.

### Prerequisites
Must have OpenCV installed with the `Video I/O` module built. 

### Build Instructions
* Place Trueface include files and `libtf.a` in `../../trueface_sdk/`
* replace `<LICENSE_CODE>` with your license code in `src/main.cpp`
* `mkdir build && cd build`
* `cmake ..`
* `make`

