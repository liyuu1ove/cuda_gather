import torch
import ctypes
import numpy as np
from functools import partial
import argparse

import performance
import sys
import os

lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.././build/lib/libmy_library.so')
lib = ctypes.CDLL(lib_path)

lib.CudaDeviceInfo
lib.hello
print("finish")