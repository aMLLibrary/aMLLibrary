#!/bin/bash

for app in output_agora_02_from_servers/*; do
  APP_NAME=$(basename $app)
  for iter in $app/*; do
    # echo python myscripts/xgboost_weights_from_pickle.py $iter
    python3 myscripts/xgboost_weights_from_pickle.py $iter
    echo
  done
done
