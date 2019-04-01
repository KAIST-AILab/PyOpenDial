import logging
from xml.etree.ElementTree import Element

import numpy as np
from multipledispatch import dispatch

from bn.distribs.density_functions.density_function import DensityFunction
from bn.values.value_factory import ValueFactory
from utils.math_utils import MathUtils
from utils.string_utils import StringUtils


class DiscreteDensityFunction(DensityFunction):
    """
    Density function defined via a set of discrete points. The density at a given
    point x is defined as the probability mass for the closest point y in the
    distribution, divided by a constant volume (used for normalisation).
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    def __init__(self, points=None):
        if isinstance(points, dict):
            """
            Creates a new discrete density function, given the set of points

            :param points: a set of (value,prob) pairs
            """

            # the set of points for the density function
            self._points = dict()  # {point: probability}
            self._points.update(points)

            # minimum distance between points
            points_keys = np.array(list(map(lambda x: np.array(x), points.keys())))
            self._min_distance = MathUtils.get_min_euclidian_distance(points_keys)
            # the volume employed for the normalisation
            self._volume = MathUtils.get_volume(self._min_distance / 2., self.get_dimensions())
        else:
            raise NotImplementedError()

    @dispatch((float, np.ndarray))
    def get_density(self, x):
        """
        Returns the density for a given point. The density is derived in two steps:
        - locating the point in the distribution that is closest to x (according to Euclidian distance)
        - dividing the probability mass for the point by the n-dimensional volume
          around this point. The radius of the ball is the half of the minimum distance
          between the points of the distribution.

        :param x: the point
        :return: the density at the point
        """
        if isinstance(x, float):
            x = np.array([x])
        closest = []
        closest_dist = np.inf
        for point in self._points.keys():
            cur_dist = MathUtils.get_distance(point, x)
            if cur_dist < closest_dist:
                closest = point
                closest_dist = cur_dist

        if closest_dist < self._min_distance / 2:
            return self._points[closest] / MathUtils.get_volume(self._min_distance / 2, self.get_dimensions())
        else:
            return 0

    @dispatch()
    def sample(self):
        """
        Samples according to the density function

        :return: the resulting sample
        """
        # TODO: check refactor > we can do better than this.
        boundary = np.random.random()
        sum = 0.
        for point in self._points.keys():
            sum += self._points[point]
            if boundary < sum:
                return point

        self.log.warning("discrete density function could not be sampled")
        raise ValueError()

    @dispatch(int)
    def discretize(self, nb_buckets):
        """
        Returns the points for this distribution.

        :param nb_buckets:
        :return: the points for this distribution
        """
        return self._points

    def __copy__(self):
        """
        Returns a copy of the density function

        :return: the copy
        """
        return DiscreteDensityFunction(self._points)

    def __str__(self):
        """
        Returns a pretty print representation of the function

        :return: the pretty print
        """
        result = ['Discrete(']
        for point in self._points.keys():
            result.append(''.join(['(', ''.join([val for val in point]), '):=', self._points[point]]))

        result.append(')')
        return ''.join(result)

    def __hash__(self):
        """
        Returns the hashcode for the function

        :return: the hashcode
        """
        return hash(self._points)

    @dispatch()
    def get_dimensions(self):
        """
        Returns the dimensionality of the distribution.

        :return: the dimensionality of the distribution
        """
        return len(list(self._points.keys())[0])

    @dispatch()
    def get_mean(self):
        """
        Returns the means of the distribution (calculated like for a categorical distribution).

        :return: the mean value.
        """
        # TODO: check refactor > we can do better than this.
        mean = [0. for _ in range(self.get_dimensions())]
        for point in self._points.keys():
            for idx, dim_value in enumerate(point):
                mean[idx] += point[idx] * self._points[point]

        return mean

    @dispatch()
    def get_variance(self):
        """
        Returns the variance of the distribution (calculated like for a categorical distribution)

        :return: the variance.
        """
        # TODO: check refactor > we can do better than this.
        variance = [0. for _ in range(self.get_dimensions())]
        mean = self.get_mean()

        for point in self._points.keys():
            for idx, dim_value in enumerate(point):
                variance[idx] += pow(point[idx] - mean[idx], 2) * self._points[point]

        return variance

    @dispatch((float, np.ndarray))
    def get_cdf(self, x):
        """
        Returns the cumulative distribution for the distribution (by counting all the
        points with a value that is lower than x).

        :param x: the point
        :return: the cumulative density function up to the point
        """
        if isinstance(x, float):
            x = np.array([x])
        if len(x) != self.get_dimensions():
            raise ValueError("Illegal dimensionality: %d != %d" % (len(x), self.get_dimensions()))

        cdf = 0.
        for point in self._points.keys():
            if MathUtils.is_lower(np.array(point), x):
                cdf += self._points[point]

        return cdf

    def generate_xml(self):
        element_list = []

        for point, prob in self._points.items():
            value_node = Element('value')
            value_node.set('prob', StringUtils.get_short_form(prob))
            value_node.text(str(ValueFactory.create(prob)))
            element_list.append(value_node)

        return element_list
