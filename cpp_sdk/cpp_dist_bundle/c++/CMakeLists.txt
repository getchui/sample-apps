cmake_minimum_required(VERSION 3.15) # 3.15 required for CMAKE_MSVC_RUNTIME_LIBRARY

option(BUILD_GPU "Build GPU sample apps" OFF)
option(BUILD_GPU_OPENCV "Build GPU sample apps which demonstrate usage of images residing in GPU memory. Must build OpenCV with CUDA + Contrib package in order to run this." OFF)

if (BUILD_GPU_OPENCV)
    set(BUILD_GPU ON)
endif()

project(sample_apps DESCRIPTION "Trueface SDK C++ Sample Apps" LANGUAGES CXX)

# Use C++ 17 Standard
set(CMAKE_CXX_STANDARD 17)

# Enable all compiler warnings, use highest level of optimization, build in release mode
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG")

if (NOT WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
endif()

if (NOT APPLE)
    # Require OpenMP for when using static library version of Trueface SDK
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

# Specify the directory containing the library and header files
set(TRUEFACE_SDK_DIR "${CMAKE_CURRENT_LIST_DIR}/../trueface_sdk")
include_directories(${TRUEFACE_SDK_DIR}/include)
link_directories(${TRUEFACE_SDK_DIR}/lib)

# Windows library has a different name
IF (WIN32)
    set(LIB_TF libtf)
ELSE()
    set(LIB_TF tf)
ENDIF()

# Need to explicitly link onnxruntime and libdl when using static library version of trueface sdk
if (WIN32)
    # Don't need to link onnxruntime for windows
    set(LINKER_LIBS ${LIB_TF} ${CMAKE_DL_LIBS})
else()
    set(LINKER_LIBS ${LIB_TF} onnxruntime ${CMAKE_DL_LIBS})
endif()

if (CMAKE_CROSSCOMPILING)
    # If we are cross compiling for arm and using the shared Trueface SDK library (libtf.so),
    # then we require ${CMAKE_DL_LIBS} as well
    list(APPEND LINKER_LIBS ${CMAKE_DL_LIBS})
endif()

# Check if token has been defined as environment variable
if(DEFINED ENV{TRUEFACE_TOKEN})
    add_definitions(-DTRUEFACE_TOKEN="$ENV{TRUEFACE_TOKEN}")
else()
    # TODO: Either define token as environment variable, or need to manually enter below
    add_definitions(-DTRUEFACE_TOKEN="YOUR_TOKEN_HERE")
endif()

# Face Recognition 1 to 1
add_executable(face_recognition_1_1 face_recognition_1_1.cpp)
target_link_libraries(face_recognition_1_1 ${LINKER_LIBS})

# Face Recognition 1 to 1 image buffer
add_executable(face_recognition_1_1_image_buffer face_recognition_1_1_image_buffer.cpp)
target_link_libraries(face_recognition_1_1_image_buffer ${LINKER_LIBS})

# Object Detection
add_executable(object_detection object_detection.cpp)
target_link_libraries(object_detection ${LINKER_LIBS})

# Spoof Detection
add_executable(spoof_detection spoof_detection.cpp)
target_link_libraries(spoof_detection ${LINKER_LIBS})

# Mask Detection
add_executable(detect_mask detect_mask.cpp)
target_link_libraries(detect_mask ${LINKER_LIBS})

# Eyeglasses Detection
add_executable(detect_eyeglasses detect_eyeglasses.cpp)
target_link_libraries(detect_eyeglasses ${LINKER_LIBS})

# Blink Detection
add_executable(blink_detection blink_detection.cpp)
target_link_libraries(blink_detection ${LINKER_LIBS})

# Face Recognition, 1 to N Matching, Enrollment
add_executable(enroll_in_database identification_1_n/enroll_in_database.cpp)
target_link_libraries(enroll_in_database ${LINKER_LIBS})

# Face Recognition, 1 to N Matching
add_executable(identification_1_n identification_1_n/identification_1_n.cpp)
target_link_libraries(identification_1_n ${LINKER_LIBS})

# Face Recognition, 1 to N Matching, multiple collections
add_executable(identification_multiple_collections identification_1_n/multiple_collections.cpp)
target_link_libraries(identification_multiple_collections ${LINKER_LIBS})

# Active spoof
add_executable(active_spoof active_spoof.cpp)
target_link_libraries(active_spoof ${LINKER_LIBS})


if (BUILD_GPU AND NOT WIN32 AND NOT APPLE)
    # GPU SDK only supported by Linux.

    # Face Recognition, batch template generation
    add_executable(batch_fr_cuda gpu_sample_apps/batch_fr_cuda.cpp)
    target_link_libraries(batch_fr_cuda ${LINKER_LIBS})
endif()

if (BUILD_GPU_OPENCV AND NOT WIN32 AND NOT APPLE)
    # GPU SDK only supported by Linux.

    # OpenCV CUDA with Contrib Packages is required for this sample app.
    find_package(OpenCV REQUIRED)

    # Face Detection, images in VRAM
    add_executable(face_recognition_image_in_vram gpu_sample_apps/face_recognition_image_in_vram.cpp)
    target_link_libraries(face_recognition_image_in_vram ${LINKER_LIBS} ${OpenCV_LIBS})
endif()
