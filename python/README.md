# Trueface SDK - Python Bindings Sample Apps
## Python Bindings Documentation
[Python Bindings Documentation](https://reference.trueface.ai/cpp/dev/latest/py/index.html)

## Getting Started
* Start by exporting your provided token as an envionrmnet variable with the key `TRUEFACE_TOKEN`.
  ex. `export TRUEFACE_TOKEN=<YOUR_TOKEN_HERE>`
* Download the appropriate version of the python bindings SDK for your platform and python version.
Next, add the directory where the python bindings library resides to your `PYTHONPATH` environment variable.
If using the GPU library, you must add the directory where the GPU library `libtf.so` resides to your `LD_LIBRARY_PATH` environment variable.
* Some sample apps may require you to have additional model files downloaded (they will throw an exception if the model file is not detected).
  The model files can be downloaded by running the scripts found in the `/download_models/` directory which comes packaged as part of the SDK download bundle.
  If you download the model files to a directory other than the run directory, you must specify the path to the directory using the `tfsdk.ConfigurationOptions.models_path` configuration option.

## Sample Apps
- [Face Recognition RTSP Server](./face_recognition_1_n_rtsp_server)
- [Active Spoof Frontend App](./active_spoof_frontend_app)