test -e 4.5.1.zip || wget https://github.com/opencv/opencv/archive/refs/tags/4.5.1.zip
test -e opencv-4.5.1 || unzip 4.5.1.zip

test -e opencv_extra_4.1.1.zip || curl -o ./opencv_extra_4.1.1.zip https://codeload.github.com/opencv/opencv_contrib/zip/4.1.1
test -e opencv_contrib-4.1.1 || unzip opencv_extra_4.1.1.zip


cd opencv-4.5.1
mkdir build
cd build

cmake -DBUILD_opencv_tracking=OFF -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.1.1/modules \
 -D WITH_CUDA=ON -D opencv_cudev=ON -D -DBUILD_opencv_cudev=ON \
 -D CPU_BASELINE=AVX -D CPU_DISPATCH=AVX2 -D CPU_DISPATCH_REQUIRE=AVX2 -D CPU_BASELINE_REQUIRE=AVX -D CMAKE_BUILD_TYPE=RELEASE \
 -D BUILD_DOCS=OFF -D BUILD_EXAMPLES=OFF -D BUILD_opencv_apps=OFF -D BUILD_opencv_python2=OFF -D BUILD_opencv_python3=OFF -D BUILD_PERF_TESTS=OFF \
 -D BUILD_TESTS=OFF -D BUILD_JPEG=ON -D BUILD_ZLIB=ON -D BUILD_PNG=ON -D BUILD_TIFF=ON -D BUILD_JASPER=OFF -D WITH_ITT=OFF -D WITH_LAPACK=OFF \
 -D WITH_TIFF=ON -D WITH_PNG=ON -D WITH_OPENCLAMDFFT=OFF -D WITH_OPENCLAMDBLAS=OFF -D WITH_VA_INTEL=OFF -D WITH_WEBP=OFF -D WITH_JASPER=OFF -D WITH_GSTREAMER=ON ..
make -j16 
sudo make install
