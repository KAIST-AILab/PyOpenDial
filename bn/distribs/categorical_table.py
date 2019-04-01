import logging
import math
from xml.etree.ElementTree import ElementTree, Element

import numpy as np
from multipledispatch import dispatch

from bn.distribs.density_functions.discrete_density_function import DiscreteDensityFunction
from bn.distribs.independent_distribution import IndependentDistribution
from bn.values.array_val import ArrayVal
from bn.values.double_val import DoubleVal
from bn.values.none_val import NoneVal
from bn.values.value import Value
from bn.values.value_factory import ValueFactory
from datastructs.assignment import Assignment
from inference.approximate.intervals import Intervals
from settings import Settings
from utils.inference_utils import InferenceUtils
from utils.math_utils import MathUtils
from utils.string_utils import StringUtils


class CategoricalTableWrapper(IndependentDistribution):
    pass


class CategoricalTable(CategoricalTableWrapper):
    """
    Representation of a categorical probability table P(X), where X is a random
    variable. Constructing a categorical table should be done via the Builder:

    builder = CategoricalTableBuilder("variable name")
    builder.addRow(...)
    categorical_table = builder.build()
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    # ===================================
    # TABLE CONSTRUCTION
    # ===================================

    def __init__(self, variable=None, head_table=None):
        if isinstance(variable, str) and isinstance(head_table, dict):
            """
            Constructs a new probability table with a mapping between head variable
            assignments and probability values. The construction assumes that the
            distribution does not have any conditional variables.

            :param variable: variable
            :param head_table: headTable
            """
            self._variable = variable
            self._table = head_table
            self._intervals = None
        else:
            raise NotImplementedError()

    @dispatch(CategoricalTableWrapper)
    def concatenate(self, other):
        """
        Concatenate the values for the two tables (assuming the two tables share the same variable).

        :param other: the table to concatenate
        :return: the table resulting from the concatenation
        """
        if self._variable != other.get_variable():
            self.log.warning("can only concatenate tables with same variable")
            raise ValueError()

        from bn.distribs.distribution_builder import CategoricalTableBuilder
        builder = CategoricalTableBuilder(self._variable)
        for s_value in self.get_values():
            for o_value in other.get_values():
                try:
                    value = s_value.concatenate(o_value)
                    builder.add_row(value, self.get_prob(s_value) * other.get_prob(o_value))
                except:
                    self.log.warning("could not concatenated the tables ", self, " and ", other)

        return builder.build()

    @dispatch(str, str)
    def modify_variable_id(self, old_id, new_id):
        """
        Modifies the distribution table by replace the old variable identifier by the new one

        :param old_id: the old identifier
        :param new_id: the new identifier
        """

        if self._variable == old_id:
            self._variable = new_id

    @dispatch(float)
    def prune_values(self, threshold):
        """
        Prunes all table values that have a probability lower than the threshold.

        :param threshold: the threshold
        :return: true if at least one value has been pruned, false otherwise
        """

        new_table = dict()
        changed = False
        for key in self._table.keys():
            prob = self._table[key]
            if prob >= threshold:
                new_table[key] = prob
            else:
                changed = True

        if changed:
            InferenceUtils.normalize(new_table)
            self._table = new_table

        self._intervals = None
        return changed

    # ===================================
    # GETTERS
    # ===================================

    @dispatch(Assignment, Value)
    def get_prob(self, condition, head):
        return super().get_prob(condition, head)

    @dispatch((str, bool, float, int, np.ndarray))
    def get_prob(self, value):
        return super().get_prob(value)

    @dispatch(Value)
    def get_prob(self, value):
        """
        Returns the probability P(val).

        :param value: the value
        :return: the associated probability, if one exists.
        """

        if value in self._table:
            return self._table[value]
        elif isinstance(value, DoubleVal) and self._is_continuous():
            # if the distribution has continuous values, search for the closest element
            to_find = value.get_double()
            closest = None
            min_distance = math.inf
            for v in self._table.keys():
                distance = abs(v.get_double() - to_find)
                if distance < min_distance:
                    closest = v
                    min_distance = distance

            return self.get_prob(closest)

        elif isinstance(value, ArrayVal) and self._is_continuous():
            to_find = value.get_array()
            closest = None
            min_distance = math.inf
            for v in self._table.keys():
                if isinstance(v, NoneVal):
                    continue

                distance = MathUtils.get_distance(v.get_array(), to_find)
                if distance < min_distance:
                    closest = v
                    min_distance = distance

            return self.get_prob(closest)

        return 0.

    @dispatch(Value)
    def has_prob(self, head):
        """
        returns true if the table contains a probability for the given assignment

        :param head: the assignment
        :return: true if the table contains a row for the assignment, false otherwise
        """
        return head in self._table

    @dispatch()
    def sample(self):
        """
        Sample a value from the distribution. If no assignment can be sampled (due to
        e.g. an ill-formed distribution), returns a none value.

        :return: the sampled assignment
        """
        if self._intervals is None:
            if len(self._table) == 0:
                self.log.warning("creating intervals for an empty table")
                raise ValueError()

            self._intervals = Intervals(self._table)

        if self._intervals.is_empty():
            self.log.warning("interval is empty, table: ", self._table)
            return ValueFactory.none()

        sample = self._intervals.sample()
        return sample

    @dispatch()
    def to_continuous(self):
        """
        Returns the continuous probability distribution equivalent to the current table

        :return: the continuous equivalent for the distribution could not be converted
        """
        if self._is_continuous():
            points = dict()

            for v in self.get_values():
                if isinstance(v, ArrayVal):
                    points[v.get_array()] = self.get_prob(v)
                elif isinstance(v, DoubleVal):
                    points[(v.get_double(),)] = self.get_prob(v)

            discrete_density_func = DiscreteDensityFunction(points)

            from bn.distribs.continuous_distribution import ContinuousDistribution
            return ContinuousDistribution(self._variable, discrete_density_func)

        raise ValueError()

    @dispatch()
    def to_discrete(self):
        """
        Returns itself.

        :return: itself
        """
        return self

    @dispatch()
    def get_variable(self):
        """
        Returns the set of variable labels used in the table

        :return: the variable labels in the table
        """
        return self._variable

    @dispatch()
    def is_empty(self):
        """
        Returns true if the table is empty (or contains only a default assignment),
        false otherwise

        :return: true if empty, false otherwise
        """
        if len(self._table) == 0:
            return True

        return len(self._table) == 1 and list(self._table.keys())[0] == ValueFactory.none()

    @dispatch(int)
    def get_n_best(self, n_best):
        """
        Returns a subset of the N values in the table with the highest probability.

        :param n_best: the number of values to select
        :return: the distribution with the subset of values
        """
        n_table = InferenceUtils.get_n_best(self._table, n_best)

        from bn.distribs.distribution_builder import CategoricalTableBuilder
        builder = CategoricalTableBuilder(self._variable)
        for v in n_table.keys():
            builder.add_row(v, n_table[v])

        return builder.build().to_discrete()

    @dispatch()
    def get_best(self):
        """
        Returns the most likely assignment of values in the table. If none could be
        found, returns an empty assignment.

        :return: the assignment with highest probability
        """
        if len(self._table) > 0:
            max_prob = -math.inf
            max_val = None
            for val in self._table.keys():
                prob = self._table[val]
                if prob > max_prob:
                    max_prob = prob
                    max_val = val
            return max_val
        else:
            self.log.warning("table is empty, cannot extract best value")
            raise ValueError()

    def __len__(self):
        """
        Returns the size of the table

        :return: the size of the table
        """
        return len(self._table)

    @dispatch()
    def get_values(self):
        """
        Returns the rows of the table.

        :return: the table rows
        """
        return set(self._table.keys())

    # ===================================
    # UTILITIES
    # ===================================

    def __hash__(self):
        """
        Returns the hashcode for the table.

        :return: the hashcode
        """
        return hash(frozenset(self._table.items()))

    def __str__(self):
        """
        Returns a string representation of the probability table

        :return: the string representation
        """
        sorted_table = InferenceUtils.get_n_best(self._table, max(len(self._table), 1))

        result = ''
        for key, value in sorted_table.items():
            result += 'P(%s=%s):=%f\n' % (self._variable, key, value)

        return result[:-1] if len(result) > 0 else result

    def __eq__(self, other):
        """
        Returns true if the object other is a categorical table with the same content
        """
        if not isinstance(other, CategoricalTable):
            return False

        other_val = other.get_values()
        if self.get_values() != other_val:
            return False

        for val in self.get_values():
            if abs(other.get_prob(val) - self.get_prob(val) > Settings.eps):
                return False

        return True

    def __copy__(self):
        """
        Returns a copy of the probability table

        :return: the copy of the table
        """
        new_table = dict()
        for v in self._table.keys():
            new_table[v] = self._table.get(v)
        return CategoricalTable(self._variable, new_table)

    @dispatch()
    def generate_xml(self):
        """
        Generates the XML representation for the table, for the document doc.

        :param doc: the XML document for which to generate the XML.
        :return: XML reprensetation
        """
        var = Element("variable")
        var.set("id", self._variable.replace("'", ""))
        for v in InferenceUtils.get_n_best(self._table, len(self._table)).keys():
            if v != ValueFactory.none():
                value_node = Element("value")
                if self._table[v] < 0.99:
                    value_node.set("prob", StringUtils.get_short_form(self._table[v]))
                value_node.text = str(v)
                var.append(value_node)

        return var

    @dispatch()
    def get_table(self):
        """
        Returns the table of values with their probability.

        :return: the table
        """
        return self._table

    # ===================================
    # PRIVATE METHODS
    # ===================================

    @dispatch()
    def _is_continuous(self):
        """
        Returns true if the table can be converted to a continuous distribution, and false otherwise.

        :return: true if convertible to continuous, false otherwise.
        """
        if len(self._table) == 0:
            return False

        for v in self.get_values():
            is_continuous = False

            if isinstance(v, ArrayVal):
                is_continuous = True

            if isinstance(v, DoubleVal):
                is_continuous = True

            if isinstance(v, NoneVal):
                is_continuous = True

            if not is_continuous:
                return False

        if len(self.get_values()) <= 1:
            return False

        return True
