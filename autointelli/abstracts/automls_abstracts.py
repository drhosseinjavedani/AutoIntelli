from abc import ABCMeta, abstractmethod

from sklearn.base import BaseEstimator, TransformerMixin


class AutoMLs(BaseEstimator, TransformerMixin):
    """Base class for creating feature selector."""

    def __init__(self, *args, **kwargs):
        """
        Class initalizer
        """
        pass

    @abstractmethod
    def data_handeler(self):
        """
        prepare data
        """
        pass
    
    @abstractmethod
    def search_space(self, *args, **kwargs):
        """
        Suggests model that being used
        """
        pass
    
    @abstractmethod
    def optimize(self, *args, **kwargs):
        """
        Optimizes the model with set of params to find best estimator
        """
        pass
    
    
    @abstractmethod
    def fit(self, *args, **kwargs):
        """
        Fit estimator using params
        """
        pass

    @abstractmethod
    def best_estimator(self, *args, **kwargs):
        """
        Calculate best estimator using params
        """
        pass

    @abstractmethod
    def transform(self, *args, **kwargs):
        """
        Return a transform
        """
        pass
    
    @abstractmethod
    def score(self, *args, **kwargs):
        """
        Score fitted model
        """
        pass




class BestEstimatorGetter(metaclass=ABCMeta):
    """Base class for creating feature selector."""

    def __init__(self, *args, **kwargs):
        """
        Class initalizer
        """
        pass

    @abstractmethod
    def best_estimator_getter(self, *args, **kwargs):
        """
        Get best estomator if any
        """
        pass


class Features(metaclass=ABCMeta):
    """Base class for creating plots for feature selector."""

    def __init__(self, *args, **kwargs):
        """
        Class initalizer
        """
        pass

    @abstractmethod
    def feature(self, *args, **kwargs):
        """
        Get list of features 
        """
        pass

    @abstractmethod
    def importance(self, *args, **kwargs):
        """
        Get list of features grades
        """
        pass

    @abstractmethod
    def plot(self, *args, **kwargs):
        """
        Creating plots 
        """
        pass

    