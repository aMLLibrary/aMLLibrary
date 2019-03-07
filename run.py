#!/usr/bin/env python3
"""
Copyright 2019 Marco Lattuada
Copyright 2019 Marjan Hosseini

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

from SequenceDataProcessing import *



def main():

    parser = argparse.ArgumentParser(description="Perform exploration of regression techniques")
    parser.add_argument('-c', "--configuration-file", help="The configuration file for the infrastructure", required=True)
    parser.add_argument('-d', "--debug", help="Enable debug messages", default=False, action="store_true")
    parser.add_argument('-s', "--seed", help="The seed", default=0)
    parser.add_argument('-o', "--output", help="The output where all the models will be stored", default="output")
    args = parser.parse_args()

    dp = SequenceDataProcessing(args)
    dp.process()

if __name__ == '__main__':
    main()
