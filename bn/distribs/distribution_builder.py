import logging

import numpy as np
from multipledispatch import dispatch

from bn.distribs.categorical_table import CategoricalTable
from bn.distribs.conditional_table import ConditionalTable
from bn.distribs.multivariate_table import MultivariateTable
from bn.distribs.single_value_distribution import SingleValueDistribution
from bn.values.value import Value
from bn.values.value_factory import ValueFactory
from datastructs.assignment import Assignment
from datastructs.value_range import ValueRange
from settings import Settings
from utils.inference_utils import InferenceUtils


class CategoricalTableBuilder:
    """
    Builder class used to construct a categorical table row-by-row. When all raws
    have been added, one can simply call build() to generate the corresponding
    distribution. When the probabilities do not sum up to 1, a default value
    "None" is added to cover the remaining probability mass.
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    # ===================================
    # TABLE CONSTRUCTION
    # ===================================

    def __init__(self, variable=None):
        if isinstance(variable, str):
            """
            Constructs a new probability table, with no values

            :param variable: the name of the random variable
            """
            self._table = dict()
            self._variable = variable
        else:
            raise NotImplementedError()

    @dispatch(Value, float)
    def add_row(self, value, prob):
        """
        Adds a new row to the probability table. If the table already contains a probability, it is erased.

        :param value: the value to add
        :param prob: the associated probability
        """

        if prob < 0.0 or prob > 1. + Settings.eps:
            return
        self._table[value] = prob

    @dispatch((str, float, bool, np.ndarray), float)
    def add_row(self, value, prob):
        """
        Adds a new row to the probability table. If the table already contains a probability, it is erased.

        :param value: the value to add (as a string)
        :param prob: the associated probability
        """
        self.add_row(ValueFactory.create(value), prob)

    @dispatch(Value, float)
    def increment_row(self, head, prob):
        """
        Increments the probability specified in the table for the given head
        assignment. If none exists, simply assign the probability.

        :param head: the head assignment
        :param prob: the probability increment
        """
        self.add_row(head, self._table.get(head, 0.) + prob)

    @dispatch(dict)
    def add_rows(self, heads):
        """
        Add a new set of rows to the probability table.

        :param heads: the mappings (head assignment, probability value)
        """
        for key, value in heads.items():
            self.add_row(key, value)

    @dispatch(Value)
    def remove_row(self, head):
        """
        Removes a row from the table.
        :param head: head assignment
        """
        self._table.pop(head)

    @dispatch()
    def normalize(self):
        InferenceUtils.normalize(self._table)

    @dispatch()
    def build(self):
        """
        Builds the categorical table based on the provided rows. If the total
        probability mass is less than 1.0, adds a default value None. If the total
        mass is higher than 1.0, normalise the table. Finally, if one single value
        is present, creates a SingleValueDistribution instead.

        :return: the distribution (CategoricalTable or SingleValueDistribution).
        """
        total_prob = sum(self._table.values())
        # TODO: check the eps value
        eps = 0.01
        if total_prob < 1. - eps:
            self.increment_row(ValueFactory.none(), max(0., 1. - total_prob))
        elif total_prob > 1. + eps:
            InferenceUtils.normalize(self._table)

        if len(self._table) == 1:
            single_value = list(self._table.keys())[0]
            return SingleValueDistribution(self._variable, single_value)
        else:
            return CategoricalTable(self._variable, self._table)

    @dispatch()
    def is_well_formed(self):
        """
        Returns true if the probability table is well-formed. The method checks
        that all possible assignments for the condition and head parts are covered
        in the table, and that the probabilities add up to 1.0f.

        :return: true if the table is well-formed, false otherwise
        """
        total_prob = sum(self._table.values())

        if total_prob < 1. - Settings.eps or total_prob > 1. + Settings.eps:
            self.log.debug("total probability is ", total_prob)
            return False

        return True

    @dispatch()
    def is_empty(self):
        """
        Returns whether the current table is empty or not

        :return: true if empty, false otherwise
        """
        return len(self._table) == 0

    @dispatch()
    def get_total_prob(self):
        """
        Returns the total probability in the table

        :return: the total probability
        """
        return sum(self._table.values())

    @dispatch()
    def get_values(self):
        """
        Returns the values included so far in the builder.

        :return: the values
        """
        # TODO: check bug >> why returning keys instead of values?
        return self._table.keys()

    @dispatch()
    def clear(self):
        """
        Clears the builder.
        """
        self._table.clear()


class ConditionalTableBuilder:
    """
    Builder class for the conditional table. The builder allows you to add rows to
    the table. Once all rows have been added, the resulting table can be created
    using the build() method.
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    def __init__(self, head_var):
        if isinstance(head_var, str):
            """
            Constructs a new conditional categorical table with the given variable name.

            :param head_var the variable name:
            """
            self._head_var = head_var
            self._table = dict()
        else:
            raise NotImplementedError()

    @dispatch(Assignment, Value, float)
    def add_row(self, condition, head, prob):
        """
        Adds a new row to the probability table, given the conditional assignment,
        the head assignment and the probability value. If the table already
        contains a probability, it is erased.

        :param condition: the conditional assignment for Y1...Yn
        :param head: the value for the head variable
        :param prob: the associated probability
        """
        if prob < 0. or prob > 1. + Settings.eps:
            self.log.warning("probability is not well-formed: %f" % prob)
            raise ValueError()

        if condition not in self._table:
            self._table[condition] = CategoricalTableBuilder(self._head_var)

        self._table[condition].add_row(head, prob)

    @dispatch(Assignment, (str, float, int, bool), float)
    def add_row(self, condition, head, prob):
        """
        Adds a new row to the probability table, given the conditional assignment,
        the head assignment and the probability value. If the table already
        contains a probability, it is erased.

        :param condition: the conditional assignment for Y1...Yn
        :param head: the value for the head variable (as a head type)
        :param prob: the associated probability
        """
        if isinstance(head, int):
            head = float(head)
        self.add_row(condition, ValueFactory.create(head), prob)

    @dispatch(Assignment, Value, float)
    def increment_row(self, condition, value, prob):
        """
        Increments the probability specified in the table for the given condition
        and head assignments. If none exists, simply assign the probability.

        :param condition: the conditional assignment
        :param value: the head assignment
        :param prob: the probability increment
        """
        if condition in self._table:
            self._table[condition].increment_row(value, prob)
        else:
            self.add_row(condition, value, prob)

    @dispatch(Assignment, dict)
    def add_rows(self, condition, subtable):
        """
        Add rows to the conditional table

        :param condition: the condition
        :param subtable: the table of values for the head distribution
        """
        builder = self._table.get(condition, None)

        if builder is None:
            builder = CategoricalTableBuilder(self._head_var)
            self._table[condition] = builder

        builder.add_rows(subtable)

    @dispatch(Assignment, Value)
    def remove_row(self, condition, head):
        """
        Removes a row from the table, given the condition and the head assignments.

        :param condition: conditional assignment
        :param head: head assignment
        :return:
        """
        if condition in self._table:
            self._table[condition].remove_row(head)
        else:
            self.log.debug("cannot remove row: condition %s is not present" % condition)
            raise ValueError()

    @dispatch()
    def fill_conditional_holes(self):
        """
        Fill the "conditional holes" of the distribution -- that is, the possible
        conditional assignments Y1,..., Yn that are not associated with any
        distribution P(X1,...,Xn | Y1,...,Yn) in the table. The method create a
        default assignment X1=None,... Xn=None with probability 1.0 for these
        cases.
        """
        possible_condition_pairs = ValueRange(self._table.keys())
        # TODO: check refactor > why 500?
        if possible_condition_pairs.get_nb_combinations() < 500:
            possible_condition_assignments = possible_condition_pairs.linearize()
            possible_condition_assignments.remove(Assignment())

            for possible_condition in possible_condition_assignments:
                if possible_condition not in self._table:
                    self.add_row(possible_condition, ValueFactory.none(), 1.)

    @dispatch()
    def is_well_formed(self):
        """
        Returns true if the probability table is well-formed. The method checks
        that all possible assignments for the condition and head parts are covered
        in the table, and that the probabilities add up to 1.0f.
        """
        possible_condition_pairs = ValueRange(self._table.keys())
        # TODO: check refactor > why 500?
        if possible_condition_pairs.get_nb_combinations() < 500:
            possible_condition_assignments = possible_condition_pairs.linearize()
            possible_condition_assignments.remove(Assignment())

            if len(possible_condition_assignments) != len(self._table.keys()) and len(possible_condition_assignments) > 1:
                # TODO: check refactor > raise exception?
                self.log.warning("number of possible conditional assignments: %d" % len(possible_condition_assignments)
                                 + ", but number of actual conditional assignments: %d" % len(self._table.keys()))
                self.log.debug("possible conditional assignments: %s" % possible_condition_assignments)
                self.log.debug("actual assignments: %s"  % self._table.keys())
                return False

        return True

    @dispatch()
    def normalize(self):
        """
        Normalises the conditional table
        """
        for assignment in self._table.keys():
            self._table[assignment].normalize()

    @dispatch()
    def build(self):
        """
        Builds the corresponding probability table. If some conditional tables
        have a total probability mass that is less than 1.0, creates a default
        None value to cover the remaining mass.

        :return: the corresponding conditional table
        """
        new_table = dict()
        for assignment in self._table.keys():
            new_table[assignment] = self._table[assignment].build()
        return ConditionalTable(self._head_var, new_table)


class MultivariateTableBuilder:
    """
    Builder for the multivariate table.
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    def __init__(self):
        self._head_vars = set()
        self._table = dict()

    @dispatch(Assignment, float)
    def add_row(self, head, prob):
        """
        Adds a new row to the probability table, assuming no conditional
        assignment. If the table already contains a probability, it is erased.

        :param head: the assignment for X1...Xn
        :param prob: the associated probability
        """
        if prob < 0. or prob > 1. + Settings.eps:
            return

        self._head_vars.update(head.get_variables())
        self._table[head] = prob

    @dispatch(Assignment, float)
    def increment_row(self, head, prob):
        """
        Increments the probability specified in the table for the given head
        assignment. If none exists, simply assign the probability.

        :param head: the head assignment
        :param prob: the probability increment
        """
        self.add_row(head, self._table.get(head, 0.) + prob)

    @dispatch(dict)
    def add_rows(self, heads):
        """
        Add a new set of rows to the probability table.

        :param heads: the mappings (head assignment, probability value)
        """
        for head in heads.keys():
            self.add_row(head, heads[head])

    @dispatch(Assignment)
    def remove_row(self, head):
        """
        Removes a row from the table.

        :param head: head assignment
        """
        del self._table[head]

    @dispatch()
    def is_well_formed(self):
        """
        Returns true if the probability table is well-formed. The method checks
        that all possible assignments for the condition and head parts are covered
        in the table, and that the probabilities add up to 1.0f.

        :return: true if the table is well-formed, false otherwise
        """
        total_prob = sum([v for v in self._table.values()])
        if total_prob < 1. - Settings.eps or total_prob > 1. + Settings.eps:
            # TODO: check refactor > raise exception?
            self.log.debug("total probability is %f" % total_prob)
            return False

        return True

    @dispatch()
    def normalize(self):
        """
        Normalises the table
        """
        InferenceUtils.normalize(self._table)

    @dispatch()
    def build(self):
        """
        Builds the multivariate table

        :return: the corresponding table
        """
        total_prob = sum([v for v in self._table.values()])
        # TODO: eps 확인 필요.
        eps = 0.01
        if total_prob < 1. - eps:
            assignment = Assignment.create_default(self._head_vars)
            self.increment_row(assignment, 1. - total_prob)
        else:
            InferenceUtils.normalize(self._table)

        return MultivariateTable(self._head_vars, self._table)
