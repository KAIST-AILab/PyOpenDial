import numpy as np
from multipledispatch import dispatch

from bn.distribs.categorical_table import CategoricalTable
from bn.distribs.independent_distribution import IndependentDistribution
from bn.values.value import Value
from bn.values.value_factory import ValueFactory
from datastructs.assignment import Assignment


class SingleValueDistribution(IndependentDistribution):
    """
    Representation of a distribution with a single value associated with a probability
    of 1.0. Although this can also be represented in a categorical table, this
    representation is much faster than operating on a full table.
    """

    def __init__(self, variable=None, value=None):
        if isinstance(variable, str) and isinstance(value, Value):
            """
            Creates a new single-value distribution

            :param variable: the variable label
            :param value: the value
            """
            self._variable = variable
            self._value = value
        elif isinstance(variable, str) and isinstance(value, str):
            """
            Creates a new single-value distribution

            :param variable: the variable label
            :param value: the value (as a string)
            """
            self._variable = variable
            self._value = ValueFactory.create(value)
        else:
            raise ValueError()

    @dispatch()
    def get_variable(self):
        """
        Returns the variable label

        :return: the variable label
        """
        return self._variable

    @dispatch(float)
    def prune_values(self, threshold):
        """
        Does nothing
        """
        return False

    @dispatch(str, str)
    def modify_variable_id(self, old_id, new_id):
        """
        Modifies the variable label

        :param old_id: the old identifier to replace
        :param new_id: the new identifier
        """
        if self._variable == old_id:
            self._variable = new_id

    @dispatch((str, bool, float, np.ndarray))
    def get_prob(self, value):
        return super().get_prob(value)

    @dispatch(Assignment, Value)
    def get_prob(self, condition, head):
        return super().get_prob(condition, head)

    @dispatch(Value)
    def get_prob(self, value):
        return 1.0 if value == self._value else 0.0

    @dispatch()
    def sample(self):
        """
        Returns the value

        :return: the value
        """
        return self._value

    @dispatch()
    def get_values(self):
        """
        Returns a singleton set with the value

        :return: a singleton set with the value
        """
        result = set()
        result.add(self._value)
        return result

    @dispatch()
    def to_continuous(self):
        """
        Returns a continuous representation of the distribution (if possible)
        """
        return self.to_discrete().to_continuous()

    @dispatch()
    def to_discrete(self):
        """
        Returns the categorical table corresponding to the distribution.
        """
        mapping = dict()
        mapping[self._value] = 1.0
        return CategoricalTable(self._variable, mapping)

    @dispatch()
    def get_best(self):
        """
        Returns the value
        """
        return self._value

    @dispatch()
    def generate_xml(self):
        """
        rns the XML representation (as a categorical table)
        """
        return self.to_discrete().generate_xml()

    def __copy__(self):
        """
        Copies the distribution
        """
        return SingleValueDistribution(self._variable, self._value)

    def __hash__(self):
        """
        Returns the distribution's hashcode

        :return: the hashcode
        """
        return hash(self._variable) - hash(self._value)

    def __str__(self):
        """
        Returns the string representation for this distribution

        :return: the string
        """
        return 'P(%s=%s)=1' % (self._variable, str(self._value))

    def __eq__(self, other):
        """
        Returns true if the assignment is identical to the one in this distribution,
        otherwise false

        :param other: the object to compare
        :return: true if equals, false otherwise
        """
        if not isinstance(other, SingleValueDistribution):
            return False

        return self._value == other._value and self._variable == other._variable
