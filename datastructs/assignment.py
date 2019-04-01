import logging
from collections import Collection, Callable
from copy import copy
from xml.etree.ElementTree import Element

import numpy as np
from multipledispatch import dispatch

from bn.values.array_val import ArrayVal
from bn.values.double_val import DoubleVal
from bn.values.value import Value
from bn.values.value_factory import ValueFactory

dispatch_namespace = dict()
# TODO: not implemented with java map entries.


class AssignmentWrapper:
    pass


class Assignment(AssignmentWrapper):
    """
    Representation of an assignment of variables (expressed via their unique
    identifiers) to specific values. The assignment is logically represented as a
    conjunction of (variable,value) pairs.

    Technically, the assignment is encoded as a map between the variable identifiers
    and their associated value. This class offers various methods are provided for
    creating, comparing and manipulating such assignments.
    """

    __slots__ = ['_map', '_cached_hash']

    # logger
    log = logging.getLogger('PyOpenDial')

    # ===================================
    # CONSTRUCTORS
    # ===================================

    def __init__(self, arg1=None, arg2=None, arg3=None, arg4=None):
        if arg1 is None and arg2 is None and arg3 is None:
            """
            Creates a new, empty assignment
            """
            self._cached_hash = 0
            # the hashmap encoding the assignment
            self._map = dict()
        elif isinstance(arg1, Assignment) and arg2 is None and arg3 is None and arg4 is None:
            assignment = arg1
            """
            Creates a copy of the assignment
    
            :param assignment: the assignment to copy
            """
            self._cached_hash = 0
            self._map = dict()
            for key, value in assignment._map.items():
                self._map[copy(key)] = copy(value)
        elif isinstance(arg1, str) and isinstance(arg2, Value) and arg3 is None and arg4 is None:
            variable, value = arg1, arg2
            """
            Creates an assignment with a single (var,value) pair
    
            :param variable: the variable label
            :param value: the value
            """
            self._cached_hash = 0
            self._map = dict()
            self._map[variable] = value
        elif isinstance(arg1, str) and isinstance(arg2, str) and arg3 is None and arg4 is None:
            variable, value = arg1, arg2
            """
            Creates a new assignment, with a single (var,value) pair
    
            :param variable: the variable label
            :param value: the value (as a string)
            """
            self._cached_hash = 0
            self._map = dict()
            self._map[variable] = ValueFactory.create(value)
        elif isinstance(arg1, str) and isinstance(arg2, float) and arg3 is None and arg4 is None:
            variable, value = arg1, arg2
            """
            Creates a new assignment, with a single (var,value) pair
    
            :param variable: the variable label
            :param value: the value (as a double)
            """
            self._cached_hash = 0
            self._map = dict()
            self._map[variable] = ValueFactory.create(value)
        elif isinstance(arg1, str) and isinstance(arg2, bool) and arg3 is None and arg4 is None:
            variable, value = arg1, arg2
            """
            Creates a new assignment, with a single (var,value) pair

            :param variable: the variable label
            :param value: the value (as a boolean)
            """
            self._cached_hash = 0
            self._map = dict()
            self._map[variable] = ValueFactory.create(value)
        elif isinstance(arg1, str) and isinstance(arg2, np.ndarray) and arg3 is None and arg4 is None:
            variable, value = arg1, arg2
            """
            Creates a new assignment, with a single (var,value) pair

            :param variable: the variable label
            :param value:
            """
            self._cached_hash = 0
            self._map = dict()
            self._map[variable] = ValueFactory.create(value)
        elif isinstance(arg1, list) and arg2 is None and arg3 is None and arg4 is None:
            assignments = arg1
            """
            Creates an assignment with a list of sub assignments
    
            :param assignments: the assignments to combine
            """
            self._cached_hash = 0
            self._map = dict()

            if isinstance(assignments[0], Assignment):
                for assignment in assignments:
                    self.add_assignment(assignment)

            elif isinstance(assignments[0], str):
                for assignment in assignments:
                    self.add_pair(assignment)
        elif isinstance(arg1, str) and arg2 is None and arg3 is None and arg4 is None:
            boolean_assignment = arg1
            """
            Creates an assignment with a single pair, given a boolean assignment such as
            "Variable" or "!Variable". If booleanAssign start with an exclamation mark,
            the value is set to false, else the value is set to true.
    
            :param boolean_assignment: the boolean assignment
            """
            self._cached_hash = 0
            self._map = dict()
            self.add_pair(boolean_assignment)
        elif isinstance(arg1, dict) and arg2 is None and arg3 is None and arg4 is None:
            pairs = arg1
            """
            Creates an assignment with a map of (var,value) pairs
    
            :param pairs: the pairs
            """
            self._cached_hash = 0
            self._map = dict()
            self.add_pairs(pairs)
        elif isinstance(arg1, Assignment) and isinstance(arg2, str) and isinstance(arg3, Value) and arg4 is None:
            assignment, variable, value = arg1, arg2, arg3
            """
            Creates an assignment from an existing one (which is copied), plus a single
            (var,value) pair
    
            :param assignment: the assignment to copy
            :param variable: the variable label
            :param value: the value
            """
            self._cached_hash = 0
            self._map = dict()
            self.add_assignment(assignment)
            self.add_pair(variable, value)
        elif isinstance(arg1, Assignment) and isinstance(arg2, str) and isinstance(arg3, str) and arg4 is None:
            assignment, variable, value = arg1, arg2, arg3
            """
            Creates an assignment from an existing one (which is copied), plus a single
            (var,value) pair
    
            :param assignment: the assignment to copy
            :param variable: the variable label
            :param value: the value
            """
            self._cached_hash = 0
            self._map = dict()
            self.add_assignment(assignment)
            self.add_pair(variable, value)
        elif isinstance(arg1, Assignment) and isinstance(arg2, str) and isinstance(arg3, float) and arg4 is None:
            assignment, variable, value = arg1, arg2, arg3
            """
            Creates an assignment from an existing one (which is copied), plus a single
            (var,value) pair
    
            :param assignment: the assignment to copy
            :param variable: the variable label
            :param value: the value
            """
            self._cached_hash = 0
            self._map = dict()
            self.add_assignment(assignment)
            self.add_pair(variable, value)
        elif isinstance(arg1, Assignment) and isinstance(arg2, str) and isinstance(arg3, bool) and arg4 is None:
            assignment, variable, value = arg1, arg2, arg3
            """
            Creates an assignment from an existing one (which is copied), plus a single
            (var,value) pair
    
            :param assignment: the assignment to copy
            :param variable: the variable label
            :param value: the value
            """
            self._cached_hash = 0
            self._map = dict()
            self.add_assignment(assignment)
            self.add_pair(variable, value)
        elif (isinstance(arg1, frozenset) or isinstance(arg1, set)) and arg2 is None and arg3 is None and arg4 is None:
            entries = arg1
            """
            Creates an assignment by adding a set of map entries
    
            :param entries: the entries to add
            """
            self._cached_hash = 0
            self._map = dict()
            for key, value in entries:
                self.add_pair(key, value)
        elif isinstance(arg1, str) and isinstance(arg2, Value) and isinstance(arg3, str) and isinstance(arg4, Value):
            variable1, value1, variable2, value2 = arg1, arg2, arg3, arg4
            """
            Creates a new assignment with two pairs of (variable,value)
    
            :param variable1: label of first variable
            :param value1: value of first variable
            :param variable2: label of second variable
            :param value2: value of second variable
            """
            self._cached_hash = 0
            self._map = dict()
            self._map[variable1] = value1
            self._map[variable2] = value2
        else:
            raise NotImplementedError()

    def __hash__(self):
        """
        Returns the hashcode associated with the assignment. The hashcode is
        calculated from the hashmap corresponding to the assignment.

        :return: the corresponding hashcode
        """
        if self._cached_hash == 0:
            result = []
            for key, value in self._map.items():
                result.append((hash(key), hash(value)))
            self._cached_hash = hash(frozenset(result))

        return self._cached_hash

    def __eq__(self, other):
        """
        Returns true if the object given as argument is an assignment identical to the
        present one

        :param other: the object to compare
        :return: true if the assignments are equal, false otherwise
        """
        if not isinstance(other, Assignment):
            return False

        return self._map == other.get_pairs()

    def __str__(self):
        """
        Returns a string representation of the assignment
        """
        if len(self._map) == 0:
            return '~'

        result = list()
        for variable, value in self._map.items():
            if value is None:
                result.append('%s=%s' % (variable, 'none'))
            elif value is True:
                result.append(variable)
            elif value is False:
                result.append('!%s' % variable)
            else:
                result.append('%s=%s' % (variable, value))
            result.append(' ^ ')

        if len(result) > 0:
            result = result[:-1]

        return ''.join(result)

    __repr__ = __str__

    @staticmethod
    @dispatch(Collection, namespace=dispatch_namespace)
    def create_default(variables):
        """
        Creates an assignment with only none values for the variable labels given as argument.

        :param args:
        :param variables: the collection of variable labels
        :return: the resulting default assignment
        """
        assignment = Assignment()
        for variable in variables:
            assignment.add_pair(variable, ValueFactory.none())

        return assignment

    @staticmethod
    @dispatch(list, namespace=dispatch_namespace)
    def create_default(variables):
        """
        Creates an assignment with only none values for the variable labels given as argument.

        :param args:
        :param variables: the collection of variable labels
        :return: the resulting default assignment
        """
        assignment = Assignment()
        for variable in variables:
            assignment.add_pair(variable, ValueFactory.none())

        return assignment

    @staticmethod
    @dispatch(Collection, str, namespace=dispatch_namespace)
    def create_one_value(variables, value):
        """
        Creates an assignment where all variable share a single common value.

        :param variables: the variables of the assignment
        :param value: the single value for all variables
        :return: the corresponding assignment
        """
        assignment = Assignment()
        for variable in variables:
            assignment.add_pair(variable, value)

        return assignment

    @staticmethod
    @dispatch(str, namespace=dispatch_namespace)
    def create_from_string(assignments_str):
        assignment = Assignment()
        assignments_str = assignments_str.split('^')
        for assignment_str in assignments_str:
            if '=' in assignment_str:
                variable = assignment_str.split('=')[0].strip()
                value = assignment_str.split('=')[1].strip()
                assignment.add_pair(variable, ValueFactory.create(value))
            elif '!' in assignment_str:
                variable = assignment_str.replace('!', '').strip()
                assignment.add_pair(variable, ValueFactory.create(False))
            else:
                variable = assignment_str.strip()
                assignment.add_pair(variable, ValueFactory.create(True))

        return assignment

    # ===================================
    # SETTERS
    # ===================================

    @dispatch(str, Value)
    def add_pair(self, variable, value):
        """
        Adds a new (var,value) pair to the assignment

        :param variable: the variable
        :param value: the value
        """
        self._map[variable] = value
        self._cached_hash = 0

    @dispatch(str, str)
    def add_pair(self, variable, value):
        """
        Adds a new (var,value) pair to the assignment

        :param variable: the variable
        :param value: the value, as a string
        """
        self._map[variable] = ValueFactory.create(value)
        self._cached_hash = 0

    @dispatch(str, float)
    def add_pair(self, variable, value):
        """
        Adds a new (var,value) pair to the assignment

        :param variable: the variable
        :param value: the value, as a float
        """
        self._map[variable] = ValueFactory.create(value)
        self._cached_hash = 0

    @dispatch(str, bool)
    def add_pair(self, variable, value):
        """
        Adds a new (var,value) pair to the assignment

        :param variable: the variable
        :param value: the value, as a boolean
        """
        self._map[variable] = ValueFactory.create(value)
        self._cached_hash = 0

    @dispatch(str, list)
    def add_pair(self, variable, value):
        """
        Adds a new (var,value) pair to the assignment

        :param variable: the variable
        :param value: the value, as a float list
        """
        self._map[variable] = ValueFactory.create(value)
        self._cached_hash = 0

    @dispatch(str)
    def add_pair(self, boolean_assignment):
        """
        Adds a new (var,value) pair as determined by the form of the argument. If the
        argument starts with an exclamation mark, the value is set to False, else the
        value is set to True.

        :param boolean_assignment: the pair to add
        """
        if not boolean_assignment.startswith("!"):
            self.add_pair(boolean_assignment, ValueFactory.create(True))
        else:
            self.add_pair(boolean_assignment[1:], ValueFactory.create(False))

    @dispatch(dict)
    def add_pairs(self, pairs):
        """
        Adds a set of (var,value) pairs to the assignment

        :param pairs: the pairs to add
        """
        self._map.update(pairs)
        self._cached_hash = 0

    @dispatch(AssignmentWrapper)
    def add_assignment(self, assignment):
        """
        Add a new set of pairs defined in the assignment given as argument (i.e. merge
        the given assignment into the present one).

        :param assignment: the assignment to merge
        """
        self.add_pairs(assignment.get_pairs())

    @dispatch(str)
    def remove_pair(self, variable):
        """
        Removes the pair associated with the var label

        :param variable: the variable to remove
        :return: the removed value
        """
        value = self._map.pop(variable, None)

        self._cached_hash = 0
        return value

    @dispatch(Collection)
    def remove_pairs(self, variables):
        """
        Remove the pairs associated with the labels

        :param variables: the variable labels to remove
        """
        for variable in variables:
            self.remove_pair(variable)

        self._cached_hash = 0

    @dispatch(Value)
    def remove_values(self, value):
        """
        Remove all pairs whose value equals toRemove

        :param value: the value to remove
        :return: the resulting assignment
        """
        # TODO: check rename > not inplace operator
        assignment = Assignment()
        for variable in self._map.keys():
            if self._map[variable] != value:
                assignment.add_pair(variable, self._map[variable])

        self._cached_hash = 0
        return assignment

    @dispatch()
    def clear(self):
        self._map.clear()
        self._cached_hash = 0

    @dispatch(Collection)
    def trim(self, variables):
        """
        Trims the assignment, where only the variables given as parameters are considered

        :param variables: the variables to consider
        """
        variables_to_remove = set(self._map) - set(variables)
        for variable in variables_to_remove:
            del self._map[variable]
        self._cached_hash = 0

    @dispatch(Collection)
    def remove_all(self, variables):
        """
        Trims the assignment, where only the variables given as parameters are considered

        :param variables: the variables to consider
        """
        for variable in variables:
            if variable in self._map:
                del self._map[variable]

        self._cached_hash = 0

    @dispatch(AssignmentWrapper)
    def intersect(self, assignment):
        """
        Returns the intersection of the two assignments

        :param assignment: second assignment
        :return: the intersection
        """
        intersection = Assignment()
        for variable in self._map.keys():
            value = self._map[variable]
            if assignment.get_value(variable) == value:
                intersection.add_pair(variable, value)

        return intersection

    @dispatch()
    def remove_primes(self):
        """
        Returns a new assignment with the primes removed.

        :return: a new assignment, without the accessory specifiers
        """
        assignment = Assignment()
        for variable in self._map.keys():
            # TODO: check bug >> need to check??
            if variable + "'" not in self._map:
                has_prime = variable[-1] == "'"
                assignment.add_pair(variable[0:-1] if has_prime else variable, self._map[variable])

        return assignment

    @dispatch(Callable)
    def filter_values(self, filter_condition):
        """
        Filter the assignment by removing all pairs that do not satisfy the given predicate

        :param filter_condition: the predicate to apply for the filtering
        """
        variables_to_remove = []
        for variable in self._map.keys():
            if not filter_condition(self._map[variable]):
                variables_to_remove.append(variable)

        for variable in variables_to_remove:
            del self._map[variable]

        self._cached_hash = 0

    @dispatch(str, str)
    def rename_var(self, old_variable_name, new_variable_name):
        """
        Returns a new assignment where the variable name is replaced.

        :param old_variable_name: old variable name
        :param new_variable_name: new variable name
        :return: the new assignment with the renamed variable
        """
        new_assignment = copy(self)
        if self.contains_var(old_variable_name):
            value = new_assignment.remove_pair(old_variable_name)
            new_assignment.add_pair(new_variable_name, value)

        return new_assignment

    @dispatch()
    def add_primes(self):
        assignment = Assignment()
        for key, value in self._map.items():
            assignment.add_pair(key + "'", value)

        return assignment

    @dispatch()
    def is_empty(self):
        """
        Returns whether the assignment is empty.

        :return: true if the assignment is empty, else false.
        """
        return len(self._map) == 0

    @dispatch()
    def get_pairs(self):
        """
        Returns the pairs of the assignment

        :return: all pairs
        """
        return self._map

    @dispatch()
    def size(self):
        """
        Returns the size (number of pairs) of the assignment

        :return: the number of pairs
        """
        return len(self._map)

    @dispatch(str)
    def contains_var(self, variable):
        """
        Returns true if the assignment contains the given variable, and false otherwise

        :param variable: the variable label
        :return: true if the variable is included, false otherwise
        """
        return variable in self._map

    @dispatch(str, Value)
    def contains_pair(self, variable, value):
        """
        Returns true if the assignment contains the given entry

        :param variable: the variable label
        :param value: the variable value
        :return: true if the assignment contains the pair, false otherwise
        """
        if variable in self._map:
            return self._map[variable] == value
        return False

    @dispatch(Collection)
    def contains_vars(self, variables):
        """
        Returns true if the assignment contains all of the given variables, and false otherwise

        :param variables: the variable labels
        :return: true if all variables are included, false otherwise
        """
        for variable in variables:
            if variable not in self._map:
                return False

        return True

    @dispatch(set)
    def contains_one_var(self, variables):
        """
        Returns true if the assignment contains at least one of the given variables,
        and false otherwise

        :param variables: the variable labels
        :return: true if at least one variable are included, false otherwise
        """
        for variable in variables:
            if variable in self._map:
                return True

        return False

    @dispatch(Collection)
    def get_trimmed(self, variables):
        """
        Returns a trimmed version of the assignment, where only the variables given as
        parameters are considered

        :param variables: the variables to consider
        :return:a new, trimmed assignment
        """
        assignment = Assignment()

        for variable, value in self._map.items():
            if variable in variables:
                assignment.add_pair(variable, value)

        assignment._cached_hash = 0
        return assignment

    @dispatch(list)
    def get_trimmed(self, variables):
        """
        Returns a trimmed version of the assignment, where only the variables given as
        parameters are considered

        :param variables: the variables to consider
        :return:a new, trimmed assignment
        """
        assignment = Assignment()

        for variable, value in self._map.items():
            if variable in variables:
                assignment.add_pair(variable, value)

        assignment._cached_hash = 0
        return assignment

    @dispatch(Collection)
    def get_pruned(self, variables):
        """
        Returns a prunes version of the assignment, where the the variables given as
        parameters are pruned out of the assignment

        :param variables: the variables to remove
        :return: a new, pruned assignment
        """
        assignment = Assignment()

        for variable, value in self._map.items():
            if variable not in variables:
                assignment.add_pair(variable, value)

        assignment._cached_hash = 0
        return assignment

    def __copy__(self):
        """
        Returns a copy of the assignment

        :return: the copy
        """
        assignment = Assignment(self)
        assignment._cached_hash = self._cached_hash
        return assignment

    @dispatch()
    def get_variables(self):
        """
        Returns the list of variables used

        :return: variables list
        """
        return set(self._map.keys())

    @dispatch()
    def get_entry_set(self):
        """
        Returns the entry set for the assignment

        :return: the entry set
        """
        return set(self._map.items())

    @dispatch(str)
    def get_value(self, variable):
        """
        Returns the value associated with the variable in the assignment, if one is
        specified. Else, returns the none value.

        :param variable: the variable
        :return: the associated value
        """
        return self._map.get(variable, ValueFactory.none())

    @dispatch()
    def get_values(self):
        """
        Returns the list of values corresponding to a subset of variables in the
        assignment (in the same order)

        :param variables: the subset of variable labels
        :return: the corresponding values
        """
        return list(self._map.values())

    @dispatch(list)
    def get_values(self, variables):
        """
        Returns the list of values corresponding to a subset of variables in the
        assignment (in the same order)

        :param variables: the subset of variable labels
        :return: the corresponding values
        """
        values = []
        for variable in variables:
            values.append(self._map.get(variable, ValueFactory.none()))

        return values

    @dispatch(AssignmentWrapper)
    def contains(self, assignment):
        """
        Returns true if the assignment contains all pairs specified in the assignment
        given as parameter (both the label and its value must match for all pairs).

        :param assignment: the assignment
        :return: true if a is contained in assignment, false otherwise
        """
        for variable in assignment.get_variables():
            if variable in self._map:
                value = assignment.get_value(variable)
                self_value = self._map[variable]
                if self_value is None and value is not None:
                    return False
                elif value != self_value:
                    return False
            else:
                return False

        return True

    @dispatch(AssignmentWrapper)
    def consistent_with(self, assignment):
        """
        Returns true if the two assignments are mutually consistent, i.e. if there is
        a label l which appears in both assignment, then their value must be equal.

        :param assignment: the second assignment
        :return: true if assignments are consistent, false otherwise
        """
        shorter_map = assignment.get_pairs() if assignment.size() < len(self._map) else self._map
        larger_map = self._map if assignment.size() < len(self._map) else assignment.get_pairs()

        for evidence_variable in shorter_map.keys():
            if evidence_variable in larger_map:
                value = larger_map[evidence_variable]
            else:
                continue

            if value != shorter_map[evidence_variable]:
                return False

        return True

    @dispatch(AssignmentWrapper, set)
    def consistent_with(self, assignment, sub_variables):
        """
        Returns true if the two assignments are mutually consistent, i.e. if there is
        a label l which appears in both assignment, then their value must be equal.
        The checks are here only done on the subset of variables subvars

        :param assignment: the second assignment
        :param sub_variables: the subset of variables to check
        :return: true if assignments are consistent, false otherwise
        """
        for sub_variable in sub_variables:
            if assignment.get_value(sub_variable) is None:
                return False

            if self._map.get(sub_variable, None) is None:
                return False

            if assignment.get_value(sub_variable) != self._map[sub_variable]:
                return False

        return True

    @dispatch()
    def is_default(self):
        """
        Returns true if the assignment only contains none values for all variables,
        and false if at least one has a different value.

        :return: true if all variables have none values, false otherwise
        """
        for variable in self._map.keys():
            if self._map[variable] != ValueFactory.none():
                return False

        return True

    @dispatch()
    def contain_continuous_values(self):
        for variable in self._map.keys():
            if isinstance(self._map[variable], DoubleVal):
                return True
            if isinstance(self._map[variable], ArrayVal):
                return True

        return False

    @dispatch()
    def generate_xml(self):
        root = Element("assignment")

        for var_id in self._map.keys():
            var = Element("variable")
            var.set("id", var_id)
            value = Element("value")
            value.text = str(self._map[var_id])
            var.append(value)
            root.append(var)
        return root
