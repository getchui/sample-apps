# Speed Benchmarks
## Build Instructions
* Export your Trueface token to your environment as `TRUEFACE_TOKEN`.
  Alternatively, open the `CMakeLists.txt` file and edit this line here: `add_definitions(-DTRUEFACE_TOKEN="YOUR_TOKEN_HERE")`.
  Replace `YOUR_TOKEN_HERE` with the license token you were provided with. If you have not yet received a token, contact sales@trueface.ai.
  This will insert your token into all the sample apps. Alternatively, you can edit each individual sample app and enter your token manually.
* `mkdir build`
* `cd build`
* `cmake ..`
* `make -j2`

If building for windows, run:
* `cmake -A x64 ..`
* `cmake --build . --config release --parallel 2`
* After building the executables, you will need to copy them from `build/Release` to `build/` before running them.
* Additionally, you must ensure that `libtf.dll` and `libpq.dll` are in the same directory as your executables.

Some benchmarks may require you to download model files.
The model files can be downloaded by running `../../download_models/download_all_models.sh`. If you download the model files to a directory other than the build directory, you must specify the path to the directory using the `Trueface::ConfigurationOptions.modelsPath` configuration option.

Note, modifying some configuration options can play a role in speed, for example:
- `Trueface::ConfigurationOptions.frModel`
- `Trueface::ConfigurationOptions.objModel`
- `Trueface::ConfigurationOptions.smallestFaceHeight`
- `Trueface::ConfigurationOptions.frVectorCompression`

Refer to the documentation page to learn more about these options. 

