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
    -s, --seed: specifies the seed to be used; it is mainly exploited in the split of the data into training, hyper-parameter selection, and validtion set. If it is not specified, seed=0 will be used making the whole process deterministic.input
    -o, --output: specifies the output directory where logs and results will be put. If the directory already exists, the script fails. This behaviour has been designed to avoid unintentinal overwriting.
    -j: specifies the maximum number of processes which can be used.
    -g, --generate-plots: enables generation of plots of type actual vs. predicted.
    -t, --self-check: enables the test of the generated regressor on the whole input set.
    -l, --details: increase the verbosity of the library. In particular, results in terms of MAPE on different sets are printed for all the built regressors and not only for the best one.
    """
    parser = argparse.ArgumentParser(description="Perform exploration of regression techniques")
    parser.add_argument('-c', "--configuration-file", help="configuration file for the infrastructure", required=True)
    parser.add_argument('-d', "--debug", help="enable debug messages", default=False, action="store_true")
    parser.add_argument('-s', "--seed", help="RNG seed", default=0)
    parser.add_argument('-o', "--output", help="output folder where all the models will be stored", default="output")
    parser.add_argument('-j', help="number of processes to be used", default=1)
    parser.add_argument('-g', "--generate-plots", help="generate plots", default=False, action="store_true")
    parser.add_argument('-t', "--self-check", help="predict the input data with the generate regressor", default=False, action="store_true")
    parser.add_argument('-l', "--details", help="print results of the single experiments", default=False, action="store_true")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    sequence_data_processor = sequence_data_processing.SequenceDataProcessing(args.configuration_file, debug=args.debug, seed=args.seed, output=args.output, j=args.j, generate_plots=args.generate_plots, self_check=args.self_check, details=args.details)
    sequence_data_processor.process()


if __name__ == '__main__':
    main()
