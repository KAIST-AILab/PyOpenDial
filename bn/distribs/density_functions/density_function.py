from multipledispatch import dispatch
import abc


class DensityFunction:
    """
    Density function for a continuous probability distribution. The density function
    can be either univariate or multivariate.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def get_density(self, x):
        """
        Returns the density value of the function at a given point

        :param x: the (possibly multivariate) point
        :return: the density value for the point calculated.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_mean(self):
        """
        Returns the mean of the density function. The size of the double array
        corresponds to the dimensionality of the function.

        :return: the density mean.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_variance(self):
        """
        Returns the variance of the density function. The size of the double array
        corresponds to the dimensionality of the function.

        :return: the density variance
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def sample(self):
        """
        Returns a sampled value given the point. The size of the double array
        corresponds to the dimensionality of the function.

        :return: the sampled value.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_dimensions(self):
        """
        Returns the dimensionality of the density function.

        :return: the dimensionality.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def discretize(self, nb_buckets):
        """
        Returns a discretised version of the density function. The granularity of the
        discretisation is defined by the number of discretisation buckets.

        :param nb_buckets: the number of discretisation buckets
        :return: a discretised probability distribution, mapping a collection of points to a probability value
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def __copy__(self):
        """
        Returns a copy of the density function

        :return: the copy
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_cdf(self, x):
        """
        Returns the cumulative probability up to the given point x.

        :param x: the (possibly multivariate) point x
        :return: the cumulative probability from 0 to x extracted
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def generate_xml(self):
        """
        Returns the XML representation (as a list of XML elements) of the density function

        :param doc: the XML document for the node
        :return: the corresponding XML elements
        """
        raise NotImplementedError()
