from multipledispatch import dispatch
import logging
import math

import numpy as np

from bn.distribs.density_functions.density_function import DensityFunction
from bn.distribs.density_functions.gaussian_density_function import GaussianDensityFunction
from utils.math_utils import MathUtils
from utils.string_utils import StringUtils


class KernelDensityFunction(DensityFunction):
    """
    Density function represented as a Gaussian kernel of data points. The distribution
    is more exactly a Product KDE (a multivariate extension of classical KDE).
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    # the kernel function
    _kernel = GaussianDensityFunction(0.0, 1.0)

    def __init__(self, points=None):
        if isinstance(points, np.ndarray):
            """
            Creates a new kernel density function with the given points

            :param points: the points
            """
            self._points = points
            if len(points) == 0:
                raise ValueError("KDE Must contain at least one point")

            self._is_bounded = self._should_be_bounded(points[0])
            self._bandwidths = self._estimate_bandwidths()
            self._sampling_deviation = self._bandwidths / math.pow(len(self._bandwidths), 2)
        else:
            raise NotImplementedError()

    @dispatch()
    def get_bandwidth(self):
        """
        Returns the bandwidth defined for the KDE

        :return: the bandwidth
        """
        return self._bandwidths

    @dispatch((float, np.ndarray))
    def get_density(self, x):
        """
        Returns the density for the given point

        :param x: the point
        :return: its density
        """
        if isinstance(x, float):
            x = np.array([x])
        density = 0.
        for point in self._points:
            density += math.exp(self._point_density(x, point))

        density /= len(self._points)

        # bounded support (cf. Jones 1993)
        if self._is_bounded:
            l = [None] * len(self._bandwidths)
            u = [None] * len(self._bandwidths)
            for idx in range(len(self._bandwidths)):
                l[idx] = (0. - x[idx]) / self._bandwidths[idx]
                u[idx] = (1. - x[idx]) / self._bandwidths[idx]

            factor = 1. / (self._kernel.get_cdf(u) - self._kernel.get_cdf(l))
            density = factor * density

        return density

    @dispatch(np.ndarray, np.ndarray)
    def _point_density(self, estimate_point, data_point):
        """
        Density of x for a single point in the KDE

        :param estimate_point: the estimate point
        :param data_point: the data point
        :return: single point density
        """
        dim = len(self._bandwidths) - 1 if self._is_bounded else len(self._bandwidths)

        sum = 0.
        for idx in range(dim):
            sum += math.log(self._kernel.get_density((estimate_point[idx] - data_point[idx]) / self._bandwidths[idx]) / self._bandwidths[idx])

        return sum

    @dispatch()
    def sample(self):
        """
        Samples from the kernel density function, first picking one of the point, and
        then deviating from it according to a Gaussian centered around it

        :return: the sampled point
        """
        # step 1 : selecting one point from the available points
        centre = self._points[np.random.randint(len(self._points))]

        # step 2: sampling a point in its vicinity (following a Gaussian)
        new_point = np.random.multivariate_normal(np.zeros_like(centre), np.identity(len(centre))) * self._sampling_deviation + centre
        total = np.sum(new_point)
        shift = min(np.min(new_point, axis=0), 0.)

        # step 3: if the density must be bounded, ensure the sum is = 1
        if self._is_bounded:
            new_point = (new_point - shift) / (total - shift * len(centre))

        return new_point

    @dispatch(int)
    def discretize(self, nb_buckets):
        """
        Returns a set of discrete values for the density function.

        :param nb_buckets: the number of values to extract
        :return: the discretised values
        """
        unique_points = np.unique(self._points, axis=0)
        nb_buckets = min(nb_buckets, len(unique_points))
        results = dict()

        for data_idx in range(nb_buckets):
            results[tuple(unique_points[data_idx, :])] = 1. / nb_buckets

        return results

    def __copy__(self):
        """
        Returns a copy of the density function

        :return: the copy
        """
        return KernelDensityFunction(self._points)

    def __str__(self):
        """
        Return a pretty print for the kernel density

        :return: the KDE string representation
        """
        result = 'KDE(mean=['
        mean = self.get_mean()
        mean_str = list()
        for mean_value in mean:
            mean_str.append(StringUtils.get_short_form(mean_value))
        result += ', '.join(mean_str)
        result += ']), std='
        average_std = 0.
        for std in self.get_variance():
            average_std += math.sqrt(std)

        result += StringUtils.get_short_form(average_std / len(mean))
        result += ') with ' + str(len(self._points)) + ' kernels'
        return result

    def __hash__(self):
        """
        Returns the hashcode for the function

        :return: the hashcode
        """
        return hash(self._points)

    @dispatch()
    def get_dimensions(self):
        """
        Returns the dimensionality of the KDE.

        :return: the dimensionality
        """
        return len(self._bandwidths)

    @dispatch()
    def get_mean(self):
        """
        Returns the mean of the KDE.

        :return: the mean
        """
        return self._points.mean(axis=0)

    @dispatch()
    def get_variance(self):
        """
        Returns the variance of the KDE.

        :return:
        """
        return self._points.var(axis=0)

    @dispatch((float, np.ndarray))
    def get_cdf(self, x):
        """
        Returns the cumulative probability distribution for the KDE.

        :param x: the point
        :return: the cumulative probability from 0 to x.
        """
        if isinstance(x, float):
            x = np.array([x])
        if len(x) != self.get_dimensions():
            raise ValueError("Illegal dimensionality: ", x.length, "!=", self.get_dimensions())
        nb_lower_points = 0
        for data_idx in range(len(self._points)):
            if MathUtils.is_lower(self._points[data_idx, :], x):
                nb_lower_points += 1

        return nb_lower_points / len(self._points)

    @dispatch()
    def _get_standard_deviations(self):
        """
        Returns the standard deviation.

        :return: the standard deviation
        """
        variance = self.get_variance()
        std = np.sqrt(variance)
        return std

    @dispatch(np.ndarray)
    def _should_be_bounded(self, point):
        """
        Returns true is the distribution is bounded to a sum == 1, and false otherwise.

        :return: true if each point should be bounded, and false otherwise
        """
        total = 0.
        for value in point:
            total += value

        if 0.99 < total < 1.01 and len(point) > 1:
            return True

        return False

    def _estimate_bandwidths(self):
        """
        Estimate the bandwidths according to Silverman's rule of thumb.
        """
        std = self._get_standard_deviations()
        silverman = 1.06 * std * math.pow(len(self._points), -1. / (4. + len(std)))
        for idx in range(len(silverman)):
            if silverman[idx] == 0.:
                silverman[idx] = 0.05

        return silverman

    def generate_xml(self):
        return GaussianDensityFunction(self._points).generate_xml()
