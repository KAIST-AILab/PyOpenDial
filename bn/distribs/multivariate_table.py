import logging
from copy import copy

import numpy as np
from multipledispatch import dispatch

from bn.distribs.categorical_table import CategoricalTable
from bn.distribs.multivariate_distribution import MultivariateDistribution
from datastructs.assignment import Assignment
from inference.approximate.intervals import Intervals
from utils.inference_utils import InferenceUtils


class MultivariateTable(MultivariateDistribution):
    """
    Representation of a multivariate categorical table P(X1,...Xn), where X1,...Xn are
    random variables.
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    # ===================================
    # TABLE CONSTRUCTION
    # ===================================

    def __init__(self, arg1=None, arg2=None):
        if isinstance(arg1, set) and isinstance(arg2, dict):
            head_vars, head_table = arg1, arg2
            """
            Constructs a new probability table with a mapping between head variable
            assignments and probability values. The construction assumes that the
            distribution does not have any conditional variables.

            :param head_vars: the variables in the table
            :param head_table: the mapping to fill the table
            """
            self._head_vars = head_vars
            self._table = head_table
            self._intervals = None
        elif isinstance(arg1, CategoricalTable) and arg2 is None:
            head_table = arg1
            """
            Constructs a new multivariate table from a univariate table.

            :param head_table: the univariate table.
            """
            self._head_vars = head_table.get_variable()
            self._table = dict()
            self._intervals = None
            for value in head_table.get_values():
                prob = head_table.get_prob(value)
                self._table[Assignment(head_table.get_variable(), value)] = prob
        elif isinstance(arg1, Assignment) and arg2 is None:
            unique_value = arg1
            """
            Create a categorical table with a unique value with probability 1.0.

            :param unique_value: the unique value for the table
            """
            self._head_vars = unique_value.get_variables()
            self._table = dict()
            self._table[unique_value] = 1.
            self._intervals = None
        else:
            raise NotImplementedError()

    @dispatch(Assignment)
    def extend_rows(self, assign):
        """
        Extend all rows in the table with the given value assignment

        :param assign: the value assignment
        """
        new_table = dict()
        for key, value in self._table.items():
            new_table[Assignment(key, assign)] = value

        self._table = new_table

    # ===================================
    # GETTERS
    # ===================================

    @dispatch()
    def get_values(self):
        """
        Returns the rows of the table

        :return: the table rows
        """
        return set(self._table.keys())

    @dispatch(Assignment)
    def get_prob(self, head):
        """
        Returns the probability P(head).

        :param head: the head assignment
        :return: the associated probability, if one exists.
        """
        if len(self._head_vars) == 0 and head.size() > 0:
            return 0.

        trimmed_head = head.get_trimmed(self._head_vars)
        if trimmed_head in self._table:
            return self._table.get(trimmed_head)

        return 0.

    @dispatch(str)
    def get_marginal(self, variable):
        """
        Returns the marginal distribution P(Xi) for a random variable Xi in X1,...Xn.

        :param variable: the variable Xi
        :return: the distribution P(Xi).
        """
        from bn.distribs.distribution_builder import CategoricalTableBuilder
        marginal = CategoricalTableBuilder(variable)

        for row in self.get_values():
            prob = self._table.get(row)
            if prob > 0.:
                marginal.add_row(row.get_value(variable), prob)

        return marginal.build()

    @dispatch(Assignment)
    def has_prob(self, head):
        """
        returns true if the table contains a probability for the given assignment

        :param head: the assignment
        :return: true if the table contains a row for the assignment, false otherwise
        """
        trimmed_head = head.get_trimmed(self._head_vars)
        return self._table.get(trimmed_head) is not None

    @dispatch()
    def sample(self):
        """
        Sample an assignment from the distribution. If no assignment can be sampled
        (due to e.g. an ill-formed distribution), returns an empty assignment.

        :return: the sampled assignment
        """
        if self._intervals is None:
            self._intervals = Intervals(self._table)

        if self._intervals.is_empty():
            raise ValueError()

        return self._intervals.sample()

    @dispatch()
    def get_variables(self):
        """
        Returns the set of variable labels used in the table

        :return: the variable labels in the table
        """
        return set(self._head_vars)

    @dispatch()
    def is_empty(self):
        """
        Returns true if the table is empty (or contains only a default assignment),
        false otherwise

        :return: true if empty, false otherwise
        """
        if len(self._table) == 0:
            return True

        if len(self._table) > 1:
            return False

        return list(self._table.keys())[0] == Assignment.create_default(self._head_vars)

    @dispatch(int)
    def get_n_best(self, n_best):
        """
        Returns a subset of the N values in the table with the highest probability.

        :param n_best: the number of values to select
        :return: the distribution with the subset of values
        """

        return MultivariateTable(self._head_vars, InferenceUtils.get_n_best(self._table, n_best))

    @dispatch()
    def get_best(self):
        """
        Returns the most likely assignment of values in the table. If none could be
        found, returns an empty assignment.

        :return: the assignment with highest probability
        """
        if len(self._table) == 0:
            self.log.warning("table is empty, cannot extract best value")
            raise ValueError()

        max_prob = -np.inf
        max_assignment = None
        for assignment in self._table.keys():
            prob = self._table[assignment]
            if prob > max_prob:
                max_prob = prob
                max_assignment = assignment

        # TODO: check refactor > there is no case of max_assignment is None
        return max_assignment if max_assignment is not None else Assignment.create_default(self._head_vars)

    # ===================================
    # UTILITIES
    # ===================================

    @dispatch(str, str)
    def modify_variable_id(self, old_id, new_id):
        """
        Modifies the variable identifiers.

        :param old_id: identifier to replace
        :param new_id: the new identifier
        """

        new_table = dict()

        for assignment in self._table.keys():
            new_assignment = copy(assignment)
            if assignment.contains_var(old_id):
                value = new_assignment.remove_pair(old_id)
                new_assignment.add_pair(new_id, value)
            new_table[new_assignment] = self._table[assignment]

        if old_id in self._head_vars:
            self._head_vars.remove(old_id)
            self._head_vars.add(new_id)

        self._table = new_table
        self._intervals = None

    def __hash__(self):
        """
        Returns the hashcode for the table.
        """
        return hash(self._table)

    def __str__(self):
        """
        Returns a string representation of the probability table

        :return: the string representation
        """
        sorted_table = InferenceUtils.get_n_best(self._table, max(len(self._table), 1))

        result = []
        for key, value in sorted_table.items():
            result.append('P(%s):=%f\n' % (str(key), value))

        return ''.join(result)[:-1] if len(result) > 0 else ''

    @dispatch(float)
    def prune_values(self, threshold):
        """
        Prunes all table values that have a probability lower than the threshold.

        :param threshold: the threshold
        """
        changed = False
        new_table = dict()
        for assignment in self._table.keys():
            prob = self._table[assignment]
            if prob >= threshold:
                new_table[assignment] = prob
            else:
                changed = True

        self._table = new_table
        return changed

    def __copy__(self):
        """
        Returns a copy of the probability table

        :return: the copy of the table
        """
        from bn.distribs.distribution_builder import MultivariateTableBuilder
        builder = MultivariateTableBuilder()
        for assignment in self._table.keys():
            builder.add_row(copy(assignment), self._table[assignment])

        return builder.build()

    def to_discrete(self):
        """
        Returns itself.
        """
        return self
