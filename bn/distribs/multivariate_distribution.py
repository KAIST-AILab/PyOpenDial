import abc

from multipledispatch import dispatch

from datastructs.assignment import Assignment


class MultivariateDistribution:
    """
    Representation of a multivariate probability distribution P(X1,...Xn), where
    X1,...Xn are random variables.
    """

    __metaclass__ = abc.ABCMeta

    @dispatch()
    @abc.abstractmethod
    def get_variables(self):
        """
        Returns the names of the random variables in the distribution

        :return: the set of variable names.
        """
        raise NotImplementedError()

    @dispatch()
    @abc.abstractmethod
    def get_values(self):
        """
        Returns the set of possible assignments for the random variables.

        :return: the set of possible assignment
        """
        raise NotImplementedError()

    @dispatch(Assignment)
    @abc.abstractmethod
    def get_prob(self, values):
        """
        Returns the probability of a particular assignment of values.

        :param values: the assignment of values to X1,...Xn.
        :return: the corresponding probability
        """
        raise NotImplementedError()

    @dispatch()
    @abc.abstractmethod
    def sample(self):
        """
        Returns a sample assignment for X1,...Xn.

        :return: the sampled assignment
        """
        raise NotImplementedError()

    @dispatch(str)
    @abc.abstractmethod
    def get_marginal(self, variable):
        """
        Returns the marginal probability distribution P(Xi) for a random variable Xi
        in X1,...Xn.

        :param variable: the random variable Xi
        :return: the marginal distribution P(Xi)
        """
        raise NotImplementedError()

    @dispatch(str, str)
    @abc.abstractmethod
    def modify_variable_id(self, old_variable_id, new_variable_id):
        """
        Modifies the variable identifier in the distribution

        :param old_variable_id: the old identifier
        :param new_variable_id: the new identifier
        """
        raise NotImplementedError()

    @dispatch()
    @abc.abstractmethod
    def to_discrete(self):
        """
        Returns a representation of the distribution as a multivariate table.

        :return: the multivariate table.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def __copy__(self):
        """
        Returns a copy of the distribution.

        :return: the copy
        """
        raise NotImplementedError()

    @dispatch(float)
    @abc.abstractmethod
    def prune_values(self, threshold):
        """
        Prunes all values assignment whose probability falls below the threshold.

        :param threshold: the threshold to apply
        :return: true if at least one value has been removed, false otherwise
        """
        raise NotImplementedError()

    @dispatch()
    @abc.abstractmethod
    def get_best(self):
        """
        Returns the value with maximum probability.

        :return: the value with maximum probability
        """
        raise NotImplementedError()
