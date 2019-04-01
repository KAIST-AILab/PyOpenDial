import abc

from multipledispatch import dispatch

from datastructs.assignment import Assignment


class UtilityFunction:
    """
    Generic interface for a utility function (also called value function), mapping
    every assignment X1, ..., Xn to a scalar utility U(X1, ...., Xn).

    Typically, at least one of these X1, ..., Xn variables consist of a decision variable.
    """
    __metaclass__ = abc.ABCMeta

    @dispatch(Assignment)
    @abc.abstractmethod
    def get_util(self, input):
        """
        Returns the utility associated with the specific assignment of values for the
        input nodes. If none exists, returns 0.0f.

        :param input: the value assignment for the input chance nodes
        :return: the associated utility
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def __copy__(self):
        """
        Creates a copy of the utility distribution

        :return: the copy
        """
        raise NotImplementedError()

    @dispatch(str, str)
    @abc.abstractmethod
    def modify_variable_id(self, old_id, new_id):
        """
        Changes the variable label

        :param old_id: the old variable label
        :param new_id: the new variable label
        """
        raise NotImplementedError()
