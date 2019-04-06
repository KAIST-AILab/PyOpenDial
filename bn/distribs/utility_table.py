import logging

from multipledispatch import dispatch

from bn.distribs.utility_function import UtilityFunction
from datastructs.assignment import Assignment
from utils.inference_utils import InferenceUtils


class UtilityTable(UtilityFunction):
    """
    Utility table that is empirically constructed from a set of samples. The table is
    defined via a mapping from assignment to utility estimates.
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    # ===================================
    # CONSTRUCTION METHODS
    # ===================================

    def __init__(self, arg1=None, arg2=None, arg3=None):
        if arg1 is None and arg2 is None and arg3 is None:
            """
            Creates a new, empty empirical utility table
            """
            self._table = dict()
            self._variables = set()
        elif isinstance(arg1, dict) and arg2 is None and arg3 is None:
            values = arg1
            """
            Constructs a new utility distribution, given the values provided as argument

            :param values: the values (assignment -> double)
            """
            self._table = dict()
            self._variables = set()
            for assignment in values.keys():
                self.set_util(assignment, values[assignment])
        else:
            raise NotImplementedError()

    @dispatch(Assignment, float)
    def increment_util(self, sample, utility):
        """
        Adds a new utility value to the estimated table

        :param sample: the sample assignment
        :param utility: the utility value for the sample
        """

        if sample not in self._table:
            self._table[Assignment(sample)] = UtilityEstimate(utility)
        else:
            self._table[Assignment(sample)].update(utility)

        self._variables.update(sample.get_variables())

    @dispatch(Assignment, float)
    def set_util(self, input, utility):
        """
        Sets the utility associated with a value assignment

        :param input: the value assignment for the input nodes
        :param utility: the resulting utility
        :return:
        """
        self._table[input] = UtilityEstimate(utility)
        self._variables.update(input.get_variables())

    @dispatch(Assignment)
    def remove_util(self, input):
        """
        Removes a utility from the utility distribution

        :param input: the assignment associated with the utility to be removed
        """

        del self._table[input]

    # ===================================
    # GETTERS
    # ===================================

    @dispatch(Assignment)
    def get_util(self, input):
        """
        Returns the estimated utility for the given assignment

        :param input: the assignment
        :return: the utility for the assignment
        """

        if input.size() != len(self._variables):
            input = input.get_trimmed(self._variables)

        if input in self._table:
            return self._table[input].get_value()

        return 0.

    @dispatch()
    def get_table(self):
        """
        Returns the table reflecting the estimated utility values for each assignment

        :return: the (assignment,utility) table
        """
        new_table = dict()
        for assignment in self._table:
            new_table[assignment] = self.get_util(assignment)

        return new_table

    @dispatch(int)
    def get_n_best(self, n_best):
        """
        Creates a table with a subset of the utility values, namely the N-best highest ones.

        :param n_best: the number of values to keep in the filtered table
        :return: the table of values, of size nbest
        """
        filtered_table = InferenceUtils.get_n_best(self.get_table(), n_best)
        return UtilityTable(filtered_table)

    @dispatch(Assignment, float)
    def get_ranking(self, input, min_diff):
        """
        Returns the ranking of the given input sorted by utility

        :param input: the input to rank
        :param min_diff: the minimum difference between utilities
        :return: the rank of the given assignment in the utility table
        """
        return InferenceUtils.get_ranking(self.get_table(), input, min_diff)

    @dispatch()
    def get_best(self):
        """
        Returns the entry with the highest utility in the table

        :return: the entry with highest utility
        """
        if len(self._table) == 0:
            return Assignment(), 0.
        return list(self.get_n_best(1).get_table().items())[0]

    @dispatch()
    def get_rows(self):
        """
        Returns the rows of the table

        :return: the rows
        """
        return set(self._table.keys())
    # ===================================
    # UTILITY METHODS
    # ===================================

    def __copy__(self):
        """
        Returns a copy of the utility table

        :return: the copy
        """
        return UtilityTable(self.get_table())

    def __hash__(self):
        """
        Returns the hashcode for the utility table

        :return: the hashcode
        """
        return hash(frozenset(self._table.items()))

    def __str__(self):
        """
        Returns a string representation for the distribution

        :return: the string representation for the table.
        """
        sorted_table = InferenceUtils.get_n_best(self.get_table(), len(self._table))

        result = []
        for key, value in sorted_table.items():
            result.append('U(%s):=%f\n' % (str(key), value))

        return ''.join(result)[:-1] if len(result) > 0 else ''

    @dispatch(str, str)
    def modify_variable_id(self, old_id, new_id):
        """
        Modifies a variable label with a new one

        :param old_id: the old variable label
        :param new_id: the new label
        :return:
        """
        new_table = dict()
        for assignment in self._table.keys():
            new_assignment = Assignment()
            for variable in assignment.get_variables():
                new_variable = new_id if variable == old_id else variable
                new_assignment.add_pair(new_variable, assignment.get_value(variable))

            new_table[new_assignment] = self._table[assignment]

        self._table = new_table


class UtilityEstimate:
    """
    Estimate of a utility value, defined by the averaged estimate itself and the
    number of values that have contributed to it (in order to correctly compute
    the average)
    """

    def __init__(self, first_value):
        if isinstance(first_value, float):
            self._average = 0.  # averaged estimate for the utility
            self._nb_values = 0  # number of values used for the average

            self.update(first_value)
        else:
            raise NotImplementedError()

    @dispatch(float)
    def update(self, new_value):
        """
        Updates the current estimate with a new value

        :param new_value: the new value
        """
        self._nb_values += 1
        self._average = self._average + (new_value - self._average) / self._nb_values

    @dispatch()
    def get_value(self):
        """
        Returns the current (averaged) estimate for the utility

        :return: the estimate
        """
        if self._nb_values > 0:
            return self._average
        else:
            return 0.

    def __str__(self):
        """
        Returns the average (as a string)
        """
        return str(self._average)

    def __hash__(self):
        """
        Returns the hashcode for the average.
        """
        return hash(self._average)
