#!/bin/env python
# Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
# See the LICENSE file for licensing terms (BSD-style).



import glob
import os
import os.path
import sys
import time
import traceback
import urllib.request, urllib.parse, urllib.error
from distutils.core import setup  # , Extension, Command

assert sys.version_info[0] == 3 and sys.version_info[1] >= 5,\
    "requires Python version 3.x"


scripts = """
    tarshards
    shardindex
    show-input
    show-model
    tarsplit
""".split()

setup(
    name='dlinputs',
    version='v0.0',
    author="Thomas Breuel",
    description="Input pipelines for deep learning.",
    packages=["dlinputs"],
    # data_files= [('share/ocroseg', models)],
    scripts=scripts,
)
