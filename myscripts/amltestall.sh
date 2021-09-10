rm -rf outputs/output_faas_test*
echo +++++ \(basic\) +++++; echo
pipenv run python ./run.py -c example_configurations/faas_test.ini \
       -o outputs/output_faas_test_1_0 -j 2
echo; echo; echo; echo +++++ SFS +++++; echo
pipenv run python ./run.py -c example_configurations/faas_test_sfs.ini \
       -o outputs/output_faas_test_2_sfs -j 2
echo; echo; echo; echo +++++ HYPEROPT +++++; echo
pipenv run python ./run.py -c example_configurations/faas_test_hyperopt.ini \
       -o outputs/output_faas_test_3_hp -j 2
echo; echo; echo; echo +++++ HYPEROPT-SFS +++++; echo
pipenv run python ./run.py -c example_configurations/faas_test_hyperopt_sfs.ini \
       -o outputs/output_faas_test_4_hp_sfs -j 2
chmod -R a+rw outputs/output_faas_test*
