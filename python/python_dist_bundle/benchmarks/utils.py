from dataclasses import dataclass
import os
import time
from typing import (List, Optional)

import tfsdk


@dataclass
class Parameters:
    do_warmup: bool = True
    num_warmup: int = 10
    batch_size: int = 16
    num_iterations: int = 200


class Stopwatch:
    def __init__(self) -> None:
        self.start_point = time.time_ns()

    def elapsedTime(self) -> float:
        return time.time_ns() - self.start_point

    def elapsedTimeMilliSeconds(self) -> float:
        now = time.time_ns()
        return now / (10 ** 6) - self.start_point / (10**6)


class SDKFactory:
    @staticmethod
    def create_basic_configuration(gpu_options: tfsdk.GPUOptions) -> tfsdk.ConfigurationOptions:
        options = tfsdk.ConfigurationOptions()

        models_path = os.getenv('MODELS_PATH')
        if models_path is None:
            models_path = './'

        options.models_path = models_path
        options.GPU_options = gpu_options

        return options

    @staticmethod
    def createSDK(gpu_options: tfsdk.GPUOptions, initialize_modules: Optional[List[str]] = None, fr_model: Optional[tfsdk.FACIALRECOGNITIONMODEL] = None, obj_model: Optional[tfsdk.OBJECTDETECTIONMODEL] = None) -> tfsdk.SDK:
        options = SDKFactory.create_basic_configuration(gpu_options)

        if fr_model is not None:
            options.fr_model = fr_model

        if obj_model is not None:
            options.obj_model = obj_model

        if initialize_modules is not None:
            for module in initialize_modules:
                if hasattr(options.initialize_module, module):
                    setattr(options.initialize_module, module, True)

        sdk = tfsdk.SDK(options)

        license = os.environ['TRUEFACE_TOKEN']
        is_valid = sdk.set_license(license)
        if is_valid is False:
            print('Error: the provided license is invalid.')
            exit(1)

        return sdk
