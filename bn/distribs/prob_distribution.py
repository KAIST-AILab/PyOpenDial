import abc

from multipledispatch import dispatch

from bn.values.value import Value
from datastructs.assignment import Assignment


class ProbDistribution:
    """
    Representation of a conditional probability distribution P(X | Y1,...Ym), where X
    is the "head" random variable for the distribution, and Y1,...Ym are the
    conditional variables
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __copy__(self):
        raise NotImplementedError()

    @dispatch()
    @abc.abstractmethod
    def get_variable(self):
        """
        Returns the name of the random variable

        :return: the name of the random variable
        """
        raise NotImplementedError()

    @dispatch()
    @abc.abstractmethod
    def get_input_variables(self):
        """
        Returns the conditional variables Y1,...Ym of the distribution

        :return: the set of conditional variables
        """
        raise NotImplementedError()

    @dispatch(Assignment, Value)
    @abc.abstractmethod
    def get_prob(self, condition, head):
        """
        Returns the probability P(head|condition), if any is specified. Else, returns 0.0f.

        :param condition: the conditional assignment for Y1,..., Ym
        :param head: the value for the random variable
        :return: the associated probability, if one exists. not be extracted
        """
        raise NotImplementedError()

    @dispatch(Assignment)
    @abc.abstractmethod
    def get_prob_distrib(self, condition):
        """
        Returns the (unconditional) probability distribution associated with the
        conditional assignment provided as argument.

        :param condition: the conditional assignment on Y1,...Ym
        :return: the independent probability distribution on X. distribution could not be extracted
        """
        raise NotImplementedError()

    @dispatch(Assignment)
    @abc.abstractmethod
    def sample(self, condition):
        """
        Returns a sample value for the distribution given a particular conditional assignment.

        :param condition: the conditional assignment for Y1,...,Ym
        :return: the sampled values for the random variable sampled
        """
        raise NotImplementedError()

    @dispatch()
    @abc.abstractmethod
    def get_values(self):
        """
         Returns the set of possible values for the distribution. If the distribution
         is continuous, the method returns a discretised set.

        :return: the values in the distribution
        """
        raise NotImplementedError()

    @dispatch(Assignment)
    @abc.abstractmethod
    def get_posterior(self, condition):
        """
        Returns a new probability distribution that is the posterior of the current
        distribution, given the conditional assignment as argument.

        :param condition: an assignment of values to (a subset of) the conditional variables
        :return: the posterior distribution
        """
        raise NotImplementedError()

    @dispatch(float)
    @abc.abstractmethod
    def prune_values(self, threshold):
        """
        Prunes values whose frequency in the distribution is lower than the given threshold.
        :param threshold: the threshold to apply for the pruning
        :return: true if at least one value has been removed, false otherwise
        """
        raise NotImplementedError()

    @dispatch(str, str)
    @abc.abstractmethod
    def modify_variable_id(self, old_id, new_id):
        """
        Changes the variable name in the distribution

        :param old_id: the old variable label
        :param new_id: the new variable label
        """
        raise NotImplementedError()
