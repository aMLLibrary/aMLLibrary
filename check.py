from model_building.predictor import Predictor

for technique in ['LRRidge','DecisionTree']:
	print(technique)
	model_path = '/home/nahuel/Documents/aml-library/a-MLLibrary/output/'+technique+'.pickle'
	obj = Predictor(regressor_file=model_path)
	try:
		print("External: "+str(obj._regressor._x_columns))
	except:
		print('_x_columns not found')
	
	try:
		print("Internal: "+str(obj._regressor.get_regressor().aml_features),end='\n\n')
	except:
		print('aml_features not found',end='\n\n')