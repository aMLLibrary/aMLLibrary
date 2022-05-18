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
import subprocess
import sys

import signal
from time import monotonic as timer
from subprocess import Popen, PIPE, TimeoutExpired


def main():
    """
    Script used to stress test the fault tolerance of the library

    Checks that interrupting the tests performed by test.py is fault tolerant
    """
    done_file_flag = os.path.join('fault_tolerance_output','done')

    start = timer()
    i = 0
    while not(os.path.exists(done_file_flag)):
        print("\n\nFault tolerance test: iteration",i+1, sep=' ',end='\n\n')
        with Popen('python3 fault_tolerance_slave.py', shell=True, stdout=PIPE, preexec_fn=os.setsid, universal_newlines=True) as process:
            try:
                output = process.communicate(timeout=100)[0]
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM) # send signal to the process group
                output = process.communicate()[0]
            else:
                print(output)
                sys.exit(1)
        i += 1
    print(timer()-start)



if __name__ == '__main__':
    main()
