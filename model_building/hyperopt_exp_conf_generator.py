import copy

import model_building.design_space
import model_building.hyperopt_experiment_configuration

class HyperoptExpConfsGenerator(model_building.design_space.ExpConfsGenerator):
    def __init__(self, wrapped_generator, campaign_configuration, seed):
        super().__init__(campaign_configuration, seed)
        self._wrapped_generator = wrapped_generator

    def generate_experiment_configurations(self, prefix, regression_inputs):
        self._logger.info("-->Generating experiments by HyperoptExpConfsGenerator")
        internal_list = self._wrapped_generator.generate_experiment_configurations(prefix, regression_inputs)
        ret_list = []
        for wrapped_point in internal_list:
            ret_list.append(model_building.hyperopt_experiment_configuration.HyperoptExperimentConfiguration(self._campaign_configuration, copy.deepcopy(regression_inputs), prefix, wrapped_point))
        self._logger.info("<--")
        return ret_list

    def __deepcopy__(self, memo):
        return HyperoptExpConfsGenerator(copy.deepcopy(self._wrapped_generator), self._campaign_configuration, self._random_generator.random())

class HyperoptSFSExpConfsGenerator(model_building.design_space.ExpConfsGenerator):
    def __init__(self, wrapped_generator, campaign_configuration, seed):
        super().__init__(campaign_configuration, seed)
        self._wrapped_generator = wrapped_generator

    def generate_experiment_configurations(self, prefix, regression_inputs):
        self._logger.info("-->Generating experiments by HyperoptSFSExpConfsGenerator")
        internal_list = self._wrapped_generator.generate_experiment_configurations(prefix, regression_inputs)
        ret_list = []
        for wrapped_point in internal_list:
            ret_list.append(model_building.hyperopt_experiment_configuration.HyperoptSFSExperimentConfiguration(self._campaign_configuration, copy.deepcopy(regression_inputs), prefix, wrapped_point))
        self._logger.info("<--")
        return ret_list

    def __deepcopy__(self, memo):
        return HyperoptSFSExpConfsGenerator(copy.deepcopy(self._wrapped_generator), self._campaign_configuration, self._random_generator.random())
