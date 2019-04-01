import logging
from xml.etree.ElementTree import Element

import numpy as np
from multipledispatch import dispatch
from scipy import stats

from bn.distribs.density_functions.density_function import DensityFunction
from bn.values.value_factory import ValueFactory


class UniformDensityFunction(DensityFunction):
    """
    (Univariate) uniform density function, with a minimum and maximum.
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    def __init__(self, min_val=None, max_val=None):
        if isinstance(min_val, float) and isinstance(max_val, float):
            """
            Creates a new uniform density function with the given minimum and maximum threshold

            :param min_val: the minimum threshold
            :param max_val: the maximum threshold
            """
            self._min_val = min_val
            self._max_val = max_val
            self._distrib = stats.uniform(min_val, max_val - min_val)
        else:
            raise NotImplementedError()

    @dispatch((float, np.ndarray))
    def get_density(self, x):
        """
        Returns the density at the given point

        :param x:
        :return: the density at the point
        """
        if isinstance(x, float):
            return self._distrib.pdf(x)
        if len(x) != 1:
            raise ValueError()

        return self._distrib.pdf(x[0])

    @dispatch()
    def sample(self):
        """
        Samples the density function

        :return: the sampled point
        """
        return self._distrib.rvs(1)

    @dispatch(int)
    def discretize(self, nb_buckets):
        """
        Returns a set of discrete values for the distribution

        :param nb_buckets: the number of buckets to employ
        :return: the discretised values and their probability mass.
        """
        result = dict()
        step = (self._max_val - self._min_val) / nb_buckets
        for i in range(nb_buckets):
            value = self._min_val + i * step + step / 2.
            result[(value,)] = 1. / nb_buckets

        return result

    @dispatch(np.ndarray)
    def get_cdf(self, x):
        """
        Returns the cumulative probability up to the given point

        :param x: the point
        :return: the cumulative probability greater than 1
        """
        if len(x) != 1:
            raise ValueError()

        x = x[0]

        return self._distrib.cdf(x)

    def __copy__(self):
        """
        Returns a copy of the density function

        :return: the copy
        """
        return UniformDensityFunction(self._min_val, self._max_val)

    def __str__(self):
        """
        Returns a pretty print for the density function

        :return: the pretty print for the density
        """
        return 'Uniform(' + str(self._min_val) + ',' + str(self._max_val) + ')'

    def __hash__(self):
        """
        Returns the hashcode for the function

        :return: the hashcode
        """
        return hash(self._max_val) - hash(self._min_val)

    @dispatch()
    def get_mean(self):
        """
        Returns the mean of the distribution

        :return: the mean
        """
        return np.array([self._distrib.mean()])

    @dispatch()
    def get_variance(self):
        """
        Returns the variance of the distribution

        :return: the variance
        """
        return np.array([self._distrib.var()])

    @dispatch()
    def get_dimensions(self):
        """
        Returns the dimensionality (constrained here to 1).

        :return: 1.
        """
        return 1

    def generate_xml(self):
        distrib_element = Element('distrib')
        distrib_element.set('type', 'uniform')

        min_element = Element('min')
        min_element.text = str(ValueFactory.create(self._min_val))
        distrib_element.append(min_element)

        max_element = Element('max')
        max_element.text = str(ValueFactory.create(self._max_val))
        distrib_element.append(max_element)

        return [distrib_element]
