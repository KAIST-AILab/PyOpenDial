from xml.etree.ElementTree import Element

from multipledispatch import dispatch
import logging

import numpy as np
from scipy import stats

from bn.distribs.density_functions.density_function import DensityFunction
from bn.values.array_val import ArrayVal
from bn.values.value_factory import ValueFactory
from utils.string_utils import StringUtils


class GaussianDensityFunction(DensityFunction):
    """
    Gaussian density function. In the multivariate case, the density function is
    currently limited to Gaussian distribution with a diagonal covariance (which are
    equivalent to the product of univariate distributions).
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    def __init__(self, arg1=None, arg2=None):
        if isinstance(arg1, np.ndarray) and isinstance(arg2, np.ndarray):
            mean, variance = arg1, arg2
            """
            Creates a new density function with the given mean and variance vector. Only
            diagonal coveriance are currently supported

            :param mean: the Gaussian mean vector
            :param variance: the variances for each dimension
            """
            if len(mean) != len(variance):
                self.log.warning("different lengths for mean and variance")
            self._mean = np.array(mean)
            self._variance = np.array(variance)
            self._std = np.sqrt(self._variance)
            self._distrib = stats.multivariate_normal(mean, variance)
        elif isinstance(arg1, float) and isinstance(arg2, float):
            mean, variance = arg1, arg2
            """
            Creates a new, univariate density function with a given mean and variance

            :param mean: the Gaussian mean
            :param variance: the variance
            :return:
            """
            if variance < 0:
                self.log.warning("variance should not be negative, but is: %f" % variance)
            self._mean = np.array([mean])
            self._variance = np.array([variance])
            self._std = np.sqrt(self._variance)
            self._distrib = stats.multivariate_normal(mean, variance)
        elif isinstance(arg1, np.ndarray) and arg2 is None:
            samples = arg1
            # GaussianDensityFunction(samples=samples)
            self._mean = samples.mean(axis=0)
            self._variance = samples.var(axis=0)
            self._std = np.sqrt(self._variance)
            self._distrib = stats.multivariate_normal(self._mean, self._variance)
        else:
            raise NotImplementedError()

    @dispatch((float, np.ndarray))
    def get_density(self, x):
        """
        Returns the density at the given point

        :param x: the point
        :return: the density at the point
        """
        return self._distrib.pdf(x)

    @dispatch()
    def sample(self):
        """
        Samples values from the Gaussian.

        :return: a sample value
        """
        sample = self._distrib.rvs(size=1)
        if isinstance(sample, float) or isinstance(sample, int):
            return np.array([sample])

        return sample

    @dispatch(int)
    def discretize(self, nb_buckets):
        """
        Returns a set of discrete values (of a size of nbBuckets) extracted from the
        Gaussian. The number of values is derived from
        Settings.NB_DISCRETISATION_BUCKETS

        :param nb_buckets: the number of buckets to employ
        :return: the set of extracted values
        """
        minima = list()
        step = list()

        for idx in range(len(self._mean)):
            minima.append(self._mean[idx] - 4. * self._std[idx])
            step.append(8. * self._std[idx] / nb_buckets)

        values = dict()  # float[] -> float

        prev_cdf = 0.
        for bucket_idx in range(nb_buckets):
            new_value = list()
            for mean_idx in range(len(self._mean)):
                new_value.append(minima[mean_idx] + bucket_idx * step[mean_idx] + step[mean_idx] / 2.)
            new_value = np.array(new_value)
            cur_cdf = self.get_cdf(new_value)
            values[tuple(new_value)] = cur_cdf - prev_cdf
            prev_cdf = cur_cdf

        return values

    @dispatch((float, np.ndarray))
    def get_cdf(self, x):
        """
        Returns the cumulative probability up to the point x

        :param x: the point
        :return: the cumulative density function up to the point
        """
        return self._distrib.cdf(x)

    def __copy__(self):
        """
        Returns a copy of the density function

        :return: the copy
        """
        return GaussianDensityFunction(np.copy(self._mean), np.copy(self._variance))

    def __str__(self):
        """
        Returns a pretty print representation of the function

        :return: the pretty print
        """
        return 'N(' + str(self._mean) + ',' + str(self._variance) + ')'

    def __hash__(self):
        """
        Returns the hashcode for the density function

        :return: the hashcode
        """
        return hash(self._mean) + hash(self._variance)

    @dispatch()
    def get_mean(self):
        """
        Returns the mean of the Gaussian.

        :return: the mean value.
        """
        return self._mean

    @dispatch()
    def get_variance(self):
        """
        Returns the variance of the Gaussian.

        :return: the variance.
        """
        return self._variance

    @dispatch()
    def get_dimensions(self):
        """
        Returns the dimensionality of the Gaussian.

        :return: the dimensionality.
        """
        return self._mean.shape[0]

    def generate_xml(self):
        distrib_element = Element('distrib')
        distrib_element.set('type', 'gaussian')

        mean_element = Element('mean')
        mean_element.text(str(ValueFactory.create(self._mean)) if len(self._mean) > 1 else str(StringUtils.get_short_form(self._mean[0])))
        distrib_element.append(mean_element)
        variance_element = Element('variance')
        variance_element.text(str(ValueFactory.create(self._variance)) if len(self._variance) > 1 else str(StringUtils.get_short_form(self._variance[0])))
        distrib_element.append(variance_element)

        return [distrib_element]

