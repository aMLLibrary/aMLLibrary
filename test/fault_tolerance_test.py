#!/usr/bin/env python3
"""
Copyright 2022 Nahuel Coliva

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

import shutil
import time

import signal
from time import monotonic as timer
from subprocess import Popen, PIPE, TimeoutExpired


def main():
    """
    Script used to stress test the fault tolerance of the library

    Checks that interrupting the tests performed by fault_tolerance_slave.py is fault tolerant
    """
    parser = argparse.ArgumentParser(description="Performs fault tolerance tests")
    parser.add_argument('-t', "--timeout", help="time elapsed between interruptions (in seconds)", type=float, default=10)
    parser.add_argument('-o', "--output", help="output folder where all the models will be stored", default="output_fault_tolerance")
    args = parser.parse_args()

    # getting the name of the directory
    # where the this file is present.
    current = os.path.dirname(os.path.realpath(__file__))
      
    # Getting the parent directory name
    # where the current directory is present.
    parent = os.path.dirname(current)

    if os.path.exists(args.output):
        output_dir = args.output
    else:
        output_dir = os.path.join(parent,args.output)

    done_file_flag = os.path.join(output_dir,'done')
    command = "python3 '"+os.path.join(current,'fault_tolerance_slave.py')+"' -o '"+output_dir+"'"

    if os.path.exists(done_file_flag):
        print(output_dir+" already exists with a complete run. Deleting and starting anew...")
        time.sleep(3)
        shutil.rmtree(output_dir)
        os.mkdir(output_dir)
    elif os.path.exists(output_dir):
        print(output_dir+" already exists. Restarting from where we left...")
        time.sleep(3)



    start = timer()
    i = 0
    while not(os.path.exists(done_file_flag)):
        print("\n\nFault tolerance test: iteration",i+1, sep=' ',end='\n\n')
        with Popen(command, shell=True, stdout=PIPE, preexec_fn=os.setsid, universal_newlines=True) as process:
            try:
                #CAUTION: timeout should be longer than the maximum time needed to train a model
                output = process.communicate(timeout=args.timeout)[0]
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM) # send signal to the process group
                output = process.communicate()[0]
            else:
                print(output)
        i += 1
    print("Fault tolerance test passed in",timer()-start, sep=' ')



if __name__ == '__main__':
    main()
