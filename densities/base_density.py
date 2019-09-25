from abc import ABC, abstractmethod


class BaseDensity(ABC):

    def __init__(self):
        self = self

    @abstractmethod
    def evaluate(self, x):
        ''' Evaluate the log-density function at the value x'''
        pass

    @abstractmethod
    def grad_evaluate(self, x):
        ''' Evaluate the gradient of the log-density function at the value x '''
        pass


    @abstractmethod
    def sample(self):
        ''' Sample from the density'''
        pass
