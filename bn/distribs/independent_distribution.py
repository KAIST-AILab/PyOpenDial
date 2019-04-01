import abc
import logging

import numpy as np
from multipledispatch import dispatch

from bn.distribs.prob_distribution import ProbDistribution
from bn.values.value import Value
from bn.values.value_factory import ValueFactory
from datastructs.assignment import Assignment


class IndependentDistribution(ProbDistribution):
    __metaclass__ = abc.ABCMeta

    log = logging.getLogger('PyOpenDial')

    @dispatch()
    def get_input_variables(self):
        """
        :return: an empty set
        """
        return set()

    @dispatch(Value)
    @abc.abstractmethod
    def get_prob(self, value):
        """
        Returns the probability P(value), if any is specified. Else, returns 0.0f.

        :param value: value the value for the random variable
        :return: the associated probability, if one exists.
        """
        raise NotImplementedError()

    @dispatch(str)
    def get_prob(self, value):
        """
        Returns the probability P(value), if any is specified. Else, returns 0.0f.

        :param value: the value for the random variable (as a string)
        :return: the associated probability, if one exists.
        """
        return self.get_prob(ValueFactory.create(value))

    @dispatch(bool)
    def get_prob(self, value):
        """
        Returns the probability P(value), if any is specified. Else, returns 0.0f.

        :param value: the value for the random variable (as a boolean)
        :return: the associated probability, if one exists.
        """
        return self.get_prob(ValueFactory.create(value))

    @dispatch((float, int))
    def get_prob(self, value):
        """
        Returns the probability P(value), if any is specified. Else, returns 0.0f.

        :param value: the value for the random variable (as a float)
        :return: the associated probability, if one exists.
        """
        return self.get_prob(ValueFactory.create(float(value)))

    @dispatch(np.ndarray)
    def get_prob(self, value):
        """
        Returns the probability P(value), if any is specified. Else, returns 0.0f.

        :param value: the value for the random variable (as a float array)
        :return: associated probability, if one exists.
        """
        return self.get_prob(ValueFactory.create(value))

    @dispatch()
    @abc.abstractmethod
    def sample(self):
        """
        Returns a sampled value for the distribution.

        :return: the sampled value
        """
        raise NotImplementedError()

    @dispatch()
    @abc.abstractmethod
    def get_values(self):
        """
        Returns a set of possible values for the distribution. If the distribution is
        continuous, assumes a discretised representation of the distribution.

        :return: the possible values for the distribution
        """
        raise NotImplementedError()

    @dispatch()
    @abc.abstractmethod
    def to_continuous(self):
        """
        Returns a continuous representation of the distribution.

        :return: the distribution in a continuous form be converted to a continuous form
        """
        raise NotImplementedError()

    @dispatch()
    @abc.abstractmethod
    def to_discrete(self):
        """
        Returns a discrete representation of the distribution

        :return:  the distribution in a discrete form.
        """
        raise NotImplementedError()

    @dispatch()
    @abc.abstractmethod
    def get_best(self):
        """
        Returns the value with maximum probability (discrete case) or the mean value
        of the distribution (continuous case)

        :return: the maximum-probability value (discrete) or the mean value (continuous)
        """
        raise NotImplementedError()

    @dispatch()
    @abc.abstractmethod
    def generate_xml(self):
        """
        Generates a XML node that represents the distribution.

        :param document: the XML node to which the node will be attached
        :return: the corresponding XML node
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def __copy__(self):
        """
        Returns a copy of the distribution.

        :return: the copied distribution
        """
        raise NotImplementedError()

    @dispatch(Assignment)
    def get_posterior(self, condition):
        """
        Returns itself

        :return: the distribution itself
        """

        return self

    @dispatch(Assignment, Value)
    def get_prob(self, condition, head):
        return self.get_prob(head)

    @dispatch(Assignment)
    def get_prob_distrib(self, condition):
        """
        Returns itself

        :return: the distribution itself
        """

        return self
