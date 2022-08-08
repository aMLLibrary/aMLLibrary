import os
import sys
import argparse

parser = argparse.ArgumentParser(description="Performs fault tolerance tests")
parser.add_argument('-t', "--timeout", help="time elapsed between interruptions (in seconds)", type=float, default=10)
parser.add_argument('-o', "--output", help="output folder where all the models will be stored", default="output_fault_tolerance")
args = parser.parse_args()

print(args)