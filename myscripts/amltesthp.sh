rm -rf outputs/output_faas_test*
pipenv run python ./run.py -c example_configurations/faas_test_hyperopt.ini \
       -o outputs/output_faas_test_3_hp
echo; echo; echo; echo
pipenv run python ./run.py -c example_configurations/faas_test_hyperopt_sfs.ini \
       -o outputs/output_faas_test_4_hp_sfs
chmod -R a+rw outputs/output_faas_test*
