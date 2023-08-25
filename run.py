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
import os

import sequence_data_processing


def main():
    """
    Main script to use the library standalone

    The main argument of this script is the configuration file which describes the experimental campaing to be performed i.e., which is the input file, which are the pre-processing steps to be performed, which technique with which hyper-parameters have to be used.
    Example of configuration files can be found in example_confiigurations directory

    Other arguments are:
    -d, --debug: enables the debug printing.
    -o, --output: specifies the output directory where logs and results will be put. If the directory already exists, the script fails. This behaviour has been designed to avoid unintentinal overwriting.
    -j: specifies the maximum number of processes which can be used.
    -l, --details: increase the verbosity of the library. In particular, results in terms of MAPE on different sets are printed for all the built regressors and not only for the best one.
    -k, --keep-temp: do not remove temporary files after successful execution
    """
    parser = argparse.ArgumentParser(description="Perform exploration of regression techniques")
    parser.add_argument('-c', "--configuration-file", help="configuration file for the infrastructure", required=True)
    parser.add_argument('-d', "--debug", help="enable debug messages", default=False, action="store_true")
    parser.add_argument('-o', "--output", help="output folder where all the models will be stored", default="output")
    parser.add_argument('-j', help="number of processes to be used", default=1)
    parser.add_argument('-l', "--details", help="print results of the single experiments", default=False, action="store_true")
    parser.add_argument('-k', "--keep-temp", help="do not remove temporary files after successful execution", default=False, action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    sequence_data_processor = sequence_data_processing.SequenceDataProcessing(args.configuration_file, debug=args.debug, output=args.output, j=args.j, details=args.details, keep_temp=args.keep_temp)
    sequence_data_processor.process()


if __name__ == '__main__':
    main()
