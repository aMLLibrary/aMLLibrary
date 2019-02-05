import Config
import json
import logging
import PreliminaryDataProcessing

logging.basicConfig(filename='./Log/log.txt', format='%(asctime)s %(levelname)s:%(message)s',
                    level=logging.DEBUG)


def dump(obj):
   for attr in dir(obj):
       if hasattr( obj, attr ):
           print( "obj.%s = %s" % (attr, getattr(obj, attr)))

def dump(obj):
   for attr in dir(obj):
       if attr=='parameters':
           print( "obj.%s = %s" % (attr, getattr(obj, attr)))


parameters = Config.Config()
parameters_dict = parameters.parameters
print(json.dumps(parameters_dict, indent=1))
logging.info(json.dumps(parameters_dict, indent=1))

