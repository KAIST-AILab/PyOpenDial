import abc

from multipledispatch import dispatch

from datastructs.assignment import Assignment


class Parameter:
    """
    Interface for a parameter associated with an effect
    """
    __metaclass__ = abc.ABCMeta

    @dispatch(Assignment)
    @abc.abstractmethod
    def get_value(self, param):
        raise NotImplementedError()

    @dispatch()
    @abc.abstractmethod
    def get_variables(self):
        raise NotImplementedError()

    @dispatch()
    @abc.abstractmethod
    def get_expression(self):
        raise NotImplementedError()
