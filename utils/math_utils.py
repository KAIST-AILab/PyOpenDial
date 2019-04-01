from multipledispatch import dispatch
import itertools
import math
import numpy as np
from collections import Collection

from bn.values.array_val import ArrayVal
from bn.values.value_factory import ValueFactory

import logging
from multipledispatch import dispatch

dispatch_namespace = dict()

def logistic_function(*values):
    if len(values) != 2:
        raise ValueError()

    value_1 = ValueFactory.create(values[0])
    value_2 = ValueFactory.create(values[1])

    if not isinstance(value_1, ArrayVal) and not isinstance(value_2, ArrayVal):
        raise ValueError()

    array_1 = value_1.get_array()
    array_2 = value_2.get_array()
    if len(array_1) != len(array_2):
        raise ValueError()

    dot_product = 0.
    for idx in range(len(array_1)):
        dot_product += (array_1[idx] * array_2[idx])

    return ValueFactory.create(1. / (1 + math.exp(-dot_product)))


class MathUtils:
    """
    Math utilities.
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    @staticmethod
    @dispatch(np.ndarray, np.ndarray, namespace=dispatch_namespace)
    def is_lower(a, b):
        """
        Returns true is all elements in the array a have a lower value than the
        corresponding elements in the array b

        :param a: the first array
        :param b: the second array
        :return: true is a is lower than b in all dimensions, and false otherwise
        """
        for idx, a_value in enumerate(a):
            if a[idx] > b[idx]:
                return False

        return True

    @staticmethod
    @dispatch((tuple, np.ndarray), (tuple, np.ndarray), namespace=dispatch_namespace)
    def get_distance(p1, p2):
        """
        Returns the Euclidian distance between two numpy arrays.

        :param p1: the first numpy array
        :param p2: the second numpy array
        :return: the distance between the two points
        """
        if isinstance(p1, tuple):
            p1 = np.array(p1)
        if isinstance(p2, tuple):
            p2 = np.array(p2)
        return np.linalg.norm(p1 - p2)

    @staticmethod
    @dispatch(Collection, namespace=dispatch_namespace)
    def get_min_euclidian_distance(points):
        """
        Returns the minimal Euclidian distance between any two pairs of points in the
        collection of points provided as argument.

        :param points: the collection of points
        :return: the minimum distance between all possible pairs of points
        """
        min_distance = math.inf

        for point1, point2 in itertools.combinations(points, 2):
            distance = MathUtils.get_distance(point1, point2)
            if distance < min_distance:
                min_distance = distance

        return min_distance

    @staticmethod
    @dispatch(float, namespace=dispatch_namespace)
    def log_gamma(x):
        """
        Returns the log-gamma value using Lanczos approximation formula

        :param x: the point
        :return: the log-gamma value for the point
        """
        return math.lgamma(x)

    @staticmethod
    @dispatch(float, namespace=dispatch_namespace)
    def gamma(x):
        """
        Returns the value of the gamma function: Gamma(x) = integral( t^(x-1) e^(-t),
        t = 0 .. infinity)

        :param x: the point
        :return: the gamma value for the point
        """
        return math.gamma(x)

    @staticmethod
    @dispatch(float, int, namespace=dispatch_namespace)
    def get_volume(radius, dim):
        """
        Returns the volume of an N-ball of a certain radius.

        :param radius: the radius
        :param dim: the number of dimensions to consider
        :return: the resulting volume
        """
        numerator = math.pow(math.pi, dim / 2.)
        denominator = MathUtils.gamma((dim / 2.) + 1)
        radius2 = math.pow(radius, dim)
        return radius2 * numerator / denominator

    @staticmethod
    def logistic_function(values, namespace=dispatch_namespace):
        return logistic_function(values)
