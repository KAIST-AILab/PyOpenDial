import logging
from copy import copy
from xml.etree.ElementTree import Element

import numpy as np
from multipledispatch import dispatch

from bn.distribs.density_functions.density_function import DensityFunction
from bn.distribs.distribution_builder import CategoricalTableBuilder
from bn.distribs.independent_distribution import IndependentDistribution
from bn.values.array_val import ArrayVal
from bn.values.double_val import DoubleVal
from bn.values.value import Value
from bn.values.value_factory import ValueFactory
from datastructs.assignment import Assignment
from settings import Settings


class ContinuousDistribution(IndependentDistribution):
    """
    Representation of a continuous probability distribution, defined by an arbitrary
    density function over a single (univariate or multivariate) variable. The
    distribution does not take any conditional assignment.
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    # ===================================
    # DISTRIBUTION CONSTRUCTION
    # ===================================

    def __init__(self, variable=None, density_func=None):
        if isinstance(variable, str) and isinstance(density_func, DensityFunction):
            """
            Constructs a new distribution with a variable and a density function

            :param variable: the variable
            :param density_func: the density function
            """
            self._variable = variable  # the variable for the distribution
            self._density_func = density_func  # density function for the distribution
            self._discrete_cache = None  # discrete equivalent of the distribution
        else:
            raise NotImplementedError()

    @dispatch(float)
    def prune_values(self, frequency_threshold):
        """
        Does nothing.
        """
        return False

    # ===================================
    # GETTERS
    # ===================================

    @dispatch()
    def sample(self):
        """
        Samples from the distribution.

        :return: the sampled (variable, value) pair
        """
        return ValueFactory.create(self._density_func.sample()) if self._density_func.get_dimensions() > 1 else ValueFactory.create(self._density_func.sample()[0])

    @dispatch((str, bool, float, np.ndarray))
    def get_prob(self, value):
        return super().get_prob(value)

    @dispatch(Assignment, Value)
    def get_prob(self, condition, head):
        return super().get_prob(condition, head)

    @dispatch(Value)
    def get_prob(self, value):
        """
        Returns the probability of the particular value, based on a discretised
        representation of the continuous distribution.

        :return: the probability value for the discretised table.
        """

        return self.to_discrete().get_prob(value)

    @dispatch()
    def get_best(self):
        """
        Returns the mean value of the distribution
        """
        return ValueFactory.create(self._density_func.get_mean())

    @dispatch()
    def to_discrete(self):
        """
        Returns a discretised version of the distribution. The number of
        discretisation buckets is defined in the configuration settings

        :return: the discretised version of the distribution
        """
        if self._discrete_cache is None:
            discretization = self._density_func.discretize(Settings.discretization_buckets)
            builder = CategoricalTableBuilder(self._variable)

            for key in discretization.keys():
                # TODO: check refactor
                key_val = np.array(key)
                val = ArrayVal(key_val) if len(key) > 1 else ValueFactory.create(key[0])
                builder.add_row(val, discretization[key])

            self._discrete_cache = builder.build().to_discrete()

        return self._discrete_cache

    @dispatch()
    def to_continuous(self):
        """
        Returns itself.
        """
        return self

    @dispatch(Value)
    def get_prob_density(self, value):
        """
        Returns the probability density for the given value

        :param value: the value (must be a DoubleVal or ArrayVal)
        :return: the resulting density
        """
        if isinstance(value, ArrayVal):
            return self._density_func.get_density(value.get_array())
        if isinstance(value, DoubleVal):
            return self._density_func.get_density(value.get_double())
        return 0.0

    @dispatch((int, float))
    def get_prob_density(self, value):
        """
        Returns the probability density for the given value

        :param value: (as a Double)
        :return: the resulting density
        """
        return self._density_func.get_density(value)

    @dispatch(np.ndarray)
    def get_prob_density(self, value):
        """
        Returns the probability density for the given value

        :param value: (as a Double array)
        :return: the resulting density
        """
        return self._density_func.get_density(value)

    @dispatch()
    def get_function(self):
        """
        Returns the density function

        :return: the density function
        """
        return self._density_func

    @dispatch()
    def get_variable(self):
        """
        Returns the variable label

        :return: the variable label
        """
        return self._variable

    @dispatch(Value)
    def get_cumulative_prob(self, value):
        """
        Returns the cumulative probability from 0 up to a given point provided in the argument.
        :param value: the value up to which the cumulative probability must be estimated.
        :return: the cumulative probability
        """
        try:
            if isinstance(value, ArrayVal):
                return self._density_func.get_cdf(value.get_array())
            elif isinstance(value, DoubleVal):
                return self._density_func.get_cdf(value.get_double())
            else:
                raise ValueError()
        except Exception as e:
            self.log.warning("exception: %s" % e)
            raise ValueError()

        return 0.0

    @dispatch(float)
    def get_cumulative_prob(self, value):
        """
        Returns the cumulative probability from 0 up to a given point provided in the argument.
        :param value: value up to which the cumulative probability must be estimated (as a double)
        :return: the cumulative probability
        """
        try:
            return self._density_func.get_cdf(value)
        except Exception as e:
            self.log.warning("exception: %s" % e)
            raise ValueError()
        return 0.0

    @dispatch(np.ndarray)
    def get_cumulative_prob(self, value):
        """
        Returns the cumulative probability from 0 up to a given point provided in the argument.

        :param value: value up to which the cumulative probability must be estimated (as an array of Doubles)
        :return: the cumulative probability
        """
        try:
            return self._density_func.get_cdf(value)
        except Exception as e:
            self.log.warning("exception: %s" % e)
        return 0.0

    @dispatch()
    def get_values(self):
        """
        Discretises the distribution and returns a set of possible values for it.

        :return: the set of discretised values for the variable
        """
        return self.to_discrete().get_values()

    # ===================================
    # UTILITY FUNCTIONS
    # ===================================

    def __copy__(self):
        """
        Returns a copy of the probability distribution

        :return: the copy
        """
        return ContinuousDistribution(self._variable, copy(self._density_func))

    def __str__(self):
        """
        Returns a pretty print of the distribution

        :return: the pretty print
        """
        return 'PDF(%s)=%s' % (self._variable, str(self._density_func))

    @dispatch(str, str)
    def modify_variable_id(self, old_id, new_id):
        """
        Modified the variable label

        :param old_id: the old variable label
        :param new_id: the new variable label
        """
        if self._variable == old_id:
            self._variable = new_id

        if self._discrete_cache is not None:
            self._discrete_cache.modify_variable_id(old_id, new_id)

    @dispatch()
    def generate_xml(self):
        """
        Returns the XML representation of the distribution

        :param doc: the document to which the XML node belongs
        :return: the corresponding node generated.
        """
        var = Element("variable")
        var.set("id", self._variable)
        for node in self._density_func.generate_xml():
            var.append(node)

        return var
