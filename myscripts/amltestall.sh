rm -rf outputs/output_faas_test*
pipenv run python ./run.py -c example_configurations/faas_test.ini \
       -o outputs/output_faas_test_1
echo; echo
pipenv run python ./run.py -c example_configurations/faas_test_hyperopt.ini \
       -o outputs/output_faas_test_2
echo; echo
pipenv run python ./run.py -c example_configurations/faas_test_sfs.ini \
       -o outputs/output_faas_test_3
echo; echo
pipenv run python ./run.py -c example_configurations/faas_test_hyperopt_sfs.ini \
       -o outputs/output_faas_test_4
chmod -R a+rw outputs/output_faas_test*
