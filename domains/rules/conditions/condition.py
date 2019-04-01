import abc
from multipledispatch import dispatch

from datastructs.assignment import Assignment


class Condition(object):
    """
    Generic interface for a condition used in a probability or utility rule.
    A condition operates on a number of (possibly underspecified) input variables, and
    can be applied to any input assignment to determine if it satisfies the condition
    or not. In addition, the condition can also produce some local groundings, for
    instance based on slots filled via string matching.
    """
    __metaclass__ = abc.ABCMeta

    @dispatch()
    @abc.abstractmethod
    def get_input_variables(self):
        raise NotImplementedError()

    @dispatch(Assignment)
    @abc.abstractmethod
    def is_satisfied_by(self, param):
        raise NotImplementedError()

    @dispatch(Assignment)
    @abc.abstractmethod
    def get_groundings(self, param):
        raise NotImplementedError()

    @dispatch()
    @abc.abstractmethod
    def get_slots(self):
        raise NotImplementedError()
