#!/usr/bin/env python3
"""
Copyright 2019 Marco Lattuada

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import os
import multiprocessing
import subprocess
import threading as th
import sys
import time

from test import main as test


def main():
    """
    Script used to stress test the fault tolerance of the library

    Checks that interrupting the tests performed by test.py is fault tolerant
    """

    
    for i in range(10):
        test_runner = th.Timer(0.0,test)
        test_runner.start()
        time.sleep(1)
        KeyboardInterrupt()
        time.sleep(20)
    




if __name__ == '__main__':
    main()
