#!/usr/bin/env python3
"""
Copyright 2021 Bruno Guindani

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


def use_command(cmd, dry_run=True):
    print(cmd)
    if not dry_run:
        subprocess.call(cmd, shell=True, executable='/bin/bash')
        print()


def main():
    desc = ("Performs multiple experiments which are located in subfolders of CONFIG_FOLDER."
            " Each subfolder may represent a device, application, or some other form of "
            "independent unit within the project. The structure of the output folders will "
            "reflect the one of CONFIG_FOLDER, i.e. it will have the same path as "
            "CONFIG_FOLDER, but with OUPUT replacing the highest-level folder. The default "
            "behaviour is performing a dry run which simply prints commands rather than "
            "executing them. To actually run the experiments, please add the -x option."
    )
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("CONFIG_FOLDER", help=("folder which contains all subfolders of .ini files"))
    parser.add_argument("-o", "--output", help="root folder for all output folders (default: outputs)", default="outputs")
    parser.add_argument("-d", "--debug", help="enable debug messages", default=False, action="store_true")
    parser.add_argument("-j", help="number of parallel processes to be used", default=1)
    parser.add_argument("-x", "--execute", help="execute the experiments instead of performing a dry run", default=False, action="store_true")
    args = parser.parse_args()

    # Get command-line options for run.py
    extra_options = []
    if args.debug:
        extra_options.append('-d')
    if int(args.j) > 1:
        extra_options.extend(('-j', args.j))
    # Get command-line options for this script
    root_config_rel_fold = args.CONFIG_FOLDER
    config_split = os.path.normpath(root_config_rel_fold).split(os.sep)
    config_split[0] = args.output
    print(config_split)
    root_output_rel_fold = os.path.join(*config_split)
    dry_run = not args.execute
    if dry_run:
        print("Performing dry run... (add -x option to actually run the experiments)")

    # Initialize output directory
    script_fold = os.path.dirname(os.path.abspath(__file__))
    root_output_abs_fold = os.path.join(script_fold, root_output_rel_fold)
    cmd_mkdir_1 = ' '.join(('mkdir', '-pv', root_output_rel_fold))
    use_command(cmd_mkdir_1, dry_run)

    # Loop over devices
    for device_name in sorted(os.listdir(root_config_rel_fold)):
        # Initialize device-specific folders
        device_config_fold = os.path.join(root_config_rel_fold, device_name)
        device_output_fold = os.path.join(root_output_rel_fold, device_name)

        # Skip non-folders
        if not os.path.isdir(device_config_fold):
            print("Skipping non-folder", device_config_fold)
            continue

        # Create output subfolder for device
        cmd_mkdir_2 = ' '.join(('mkdir', '-v', device_output_fold))
        use_command(cmd_mkdir_2, dry_run)

        # Loop over experiments i.e. configuration files
        for config_name in sorted(os.listdir(device_config_fold)):
            if not config_name.endswith('.ini'):
                print("Skipping non .ini file", config_name)
                continue
            exper_config_path = os.path.join(device_config_fold, config_name)
            config_name_no_ext = config_name[:-4]
            exper_output_path = os.path.join(device_output_fold, config_name_no_ext)
            cmd_run = ' '.join(['python3', 'run.py'] + extra_options +
                               ['-c', exper_config_path,
                                '-o', exper_output_path])
            use_command(cmd_run, dry_run)


if __name__ == '__main__':
    main()
