from bn.values.value import Value
from datastructs.assignment import Assignment
from utils.inference_utils import InferenceUtils

import logging
from multipledispatch import dispatch
from collections import Collection

class ValueRangeWrapper:
    pass


class ValueRange(ValueRangeWrapper):
    """
    Representation of a range of alternative values for a set of variables.
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    def __init__(self, arg1=None, arg2=None):
        if arg1 is None and arg2 is None:
            """
            Constructs a new, empty range of values.
            """
            self._range = dict()

        elif isinstance(arg1, set) and arg2 is None:
            assignments = arg1
            """
            Creates a value range out of a set of value assignments
    
            :param assignments: the assignments specifying the possible (variable,value) pairs
            """
            self._range = dict()
            for assignment in assignments:
                self.add_assign(assignment)

        elif isinstance(arg1, ValueRangeWrapper) and isinstance(arg2, ValueRangeWrapper):
            range1, range2 = arg1, arg2
            """
            Constructs a new range that is the union of two existing ranges
    
            :param range1: the first range of values
            :param range2: the second range of values
            """
            self._range = dict()
            self.add_range(range1)
            self.add_range(range2)

        elif isinstance(arg1, dict) and arg2 is None:
            range = arg1
            """
            Constructs a range of values based on the mapping between variables and sets of values
    
            :param range: the range (as a dict)
            """
            self._range = range

        else:
            raise NotImplementedError()

    @dispatch(str, Value)
    def add_value(self, variable, value):
        """
        Adds a value for the variable in the range.

        :param variable: the variable
        :param value: the value
        """
        if variable not in self._range:
            value_set = set()
            self._range[variable] = value_set
        else:
            value_set = self._range[variable]

        value_set.add(value)

    @dispatch(str, Collection)
    def add_values(self, variable, values):
        """
        Adds a set of values for the variable

        :param variable: the variable
        :param values: the value
        """
        for value in values:
            self.add_value(variable, value)

    @dispatch(Assignment)
    def add_assign(self, assignment):
        """
        Adds the values defined in the assignment to the range

        :param assignment: the value assignments
        """
        for variable in assignment.get_variables():
            self.add_value(variable, assignment.get_value(variable))

    @dispatch(ValueRangeWrapper)
    def add_range(self, range):
        """
        Adds the range of values to the existing one.

        :param range: the new range
        """
        for variable in range.get_variables():
            self.add_values(variable, range.get_values(variable))

    @dispatch()
    def linearize(self):
        """
        Extracts all alternative assignments of values for the variables in the range.
        This operation can be computational expensive, use with caution.

        :return: the set of alternative assignments
        """
        if len(self._range) == 1:
            item_key = list(self._range.keys())[0]
            result = set()
            for item_value in self._range[item_key]:
                result.add(Assignment(item_key, item_value))
            return result

        return InferenceUtils.get_all_combinations(self._range)

    @dispatch()
    def get_nb_combinations(self):
        """
        Returns the estimated number (higher bound) of combinations for the value range.

        :return: the higher bound on the number of possible combinations
        """
        result = 1
        for value_set in self._range.values():
            result *= len(value_set)

        return result

    @dispatch()
    def get_variables(self):
        """
        Returns the set of variables with a non-empty range of values

        :return: the set of variables
        """
        return set(self._range.keys())

    @dispatch(str)
    def get_values(self, variable):
        """
        Returns the set of values for the variable in the range (if defined). If the
        variable is not defined in the range, returns null

        :param variable: the variable
        :return: its set of alternative values
        """
        return self._range[variable]

    def __str__(self):
        """
        Returns a string representation for the range
        """
        value_set_str_list = []
        for variable, value_set in self._range.items():
            value_set_str_list.append(variable + '=[' + ', '.join([str(value) for value in value_set]) + ']')

        return '{' + ', '.join(value_set_str_list) + '}'

    def __hash__(self):
        """
        Returns the hashcode for the range of values
        """
        return hash(self._range) - 1

    @dispatch()
    def is_empty(self):
        """
        Returns true if the range is empty (contains no variables).

        :return: true if empty, else false.
        """
        return len(self._range) == 0

    @dispatch(ValueRangeWrapper)
    def intersect_range(self, other):
        """
        Intersects the range with the existing one (only retains the values defined in
        both ranges).

        :param other: the range to intersect with the existing one
        """
        for variable in other.get_variables():
            if variable in self._range:
                self._range[variable].retain_all(other.get_values(variable))
            else:
                self.add_values(variable, other.get_values(variable))

    @dispatch(set)
    def remove_variables(self, variables):
        """
        Remove the variables from the value range

        :param variables: the variables to remove
        """
        for variable in variables:
            del self._range[variable]

    @dispatch(set)
    def get_sub_range(self, slots):
        result = ValueRange(self._range)
        for variable in self._range.keys():
            if variable not in slots:
                del result._range[variable]

        return result
