###
# Design space
#

class DesignSpace:
     points_setup = None;

    def explore(self):
        pass

class MultiPoints(DesignSpace):

    design_space_explorator = None;

    def explore(self):
        while design_space_explorator.has_points():
            points = design_space_explorator.get_next_points()
            #Parallel
            for point in points:
                point.explore()

class SinglePoint(DesignSpace):

    def explore(self):
        model_builder = points_setup.compute()[0]
        filtered_data = self.filter(data)
        model_builder.build(filtered_data)

class RepeatPoint(DesignSpace):
    point_to_be_repeated = None

    def explore(self):
        for i in range(0, repetion):
            point_to_be_repeated.explore()

class KFoldValidation(DesignSpace):
    point_to_be_exlored:
    def explore(self):
        for in in range(0, kfolds):
            point_to_be_explored.explore()


###
# Design space exploration
#

class DesignSpaceExplorator:
    def get_next_points(n_threads):
        pass

class GridSearch(DesignSpaceExplorator):
    def get_next_points(n_threads):
        #return n_threads points

class RandomSearch(DesignSpaceExplorator):
    def get_next_points(n_threads):
        #return n_threads points

class FeatureSelector(DesignSpaceExplorator):
    def get_next_points(n_threads):



###
# Structure used to describe the points to be analyzed
#

class PointsSetup:
    #For each technique the set of points to be considered
    technique_points_setup = None

class TechniquePointsSetup:
    #abstract class storing generalizing setup points for single technique

class LinearRegressionPointsSetup(PointsSetup):
    #it stores the setup for one or more linear regression



###
# Class to filter columns
#
class Filter:
    #Class for filtering data (features)
    def filter(data):
        #do nothing
        return data

class DegreeFilter(Filter):
    #Class removing all columns with degree larger than n
    def filter(data):
        return remove_filtered_data


###
# Data preparation
#
class DataPreparation:
    #Class for preparing data
    def prepare(data):
        #do nothing
        return data:

class GPUDataPreparation(DataPreparation):
    def prepare(data):

class SparkDataPreparation(DataPreparation):
    def prepare(data):

class Ernest(SparkDataPreparation):
    def prepare(data):

class HybridDataPreparation(DataPreparation):
    def prepare(data):

class ScaleDataPreparation(DataPreparation):
    def prepare(data):

class FeatureSelectionDataPreparation(DataPreparation):


###
# Data split
#

class DataSplit:
    def split():

class HoldOnDataSplit(DataSplit):
    def split():

class CVDataSplit(DataSplit):
    def split():


###
# Main class
#
class Exploration:
    data_preparations = []
    design_space = None
    evaluator = None
