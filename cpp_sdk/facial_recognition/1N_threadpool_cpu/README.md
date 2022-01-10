# Trueface SDK C++ Sample App
## Facial Recognition - 1:N Identification with Thread pool (CPU)
This sample apps demonstrates how to build a scalable architecture for 1 to N search using a thread pool with CPU only.
For the sake of demonstration, the entire app is contained within a single process (which is not very scalable in itself).
In order to make it truly scalable, separate each worker thread into it's own microservice process, and use a Message Queue system
such as [RabbitMQ](https://github.com/CopernicaMarketingSoftware/AMQP-CPP) to facilitate communication between the processes. This allows for the microservices to be distributed across multiple devices.
Therefore more services can be spun up (each running on their own CPU) as the demand increases.
This app shows the general approach you can take - primarily, how to pass data between services (which are simulated by different threads) 
and which SDK functions to call. 

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
* Place Trueface header files in `../../trueface_sdk/include/` and trueface libraries in `../../trueface_sdk/lib/`
* replace `<LICENSE_CODE>` with your license code in `src/main.cpp`
* `mkdir build && cd build`
* `cmake ..`
* `make`

