rm -rf outputs/output_faas_test*
echo +++++ \(basic\) +++++; echo
pipenv run python ./run.py -c example_configurations/faas_test.ini \
       -o outputs/output_faas_test_1_0
echo; echo; echo; echo +++++ HYPEROPT +++++; echo
pipenv run python ./run.py -c example_configurations/faas_test_hyperopt.ini \
       -o outputs/output_faas_test_2_hp
echo; echo; echo; echo +++++ SFS +++++; echo
pipenv run python ./run.py -c example_configurations/faas_test_sfs.ini \
       -o outputs/output_faas_test_3_sfs
echo; echo; echo; echo +++++ HYPEROPT-SFS +++++; echo
pipenv run python ./run.py -c example_configurations/faas_test_hyperopt_sfs.ini \
       -o outputs/output_faas_test_4_hp_sfs
chmod -R a+rw outputs/output_faas_test*
