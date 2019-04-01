import logging
from collections import Collection

from multipledispatch import dispatch

from datastructs.assignment import Assignment
from copy import copy


class DoubleFactor:
    """
    Double factor, combining probability and utility distributions
    """
    log = logging.getLogger('PyOpenDial')

    def __init__(self, arg1=None):
        if arg1 is None:
            """
            Creates a new, empty factor, and a set of head variables
            """
            self._matrix = dict()
        elif isinstance(arg1, DoubleFactor):
            existing_factor = arg1
            """
            Creates a new factor out of an existing one

            :param existing_factor: the existing factor
            """
            self._matrix = dict()
            for key, value in existing_factor._matrix.items():
                self._matrix[copy(key)] = copy(value)
        else:
            raise NotImplementedError("UNDEFINED PARAMETERS")

    def __copy__(self):
        return DoubleFactor(self)

    def __str__(self):
        results = []
        for assignment in self._matrix.keys():
            results.append('P(')
            results.append(str(assignment))
            results.append(')=')
            results.append(self._matrix[assignment][0])

            if self._matrix[assignment][1] != 0.:
                results.append(' and U(')
                results.append(str(assignment))
                results.append(')=')
                results.append(self._matrix[assignment][1])

            results.append('\n')

        return ''.join(results)

    def __len__(self):
        return len(self._matrix)

    @dispatch(Assignment, float, float)
    def add_entry(self, assignment, prob_value, utility_value):
        """
        Add entry to this double factor.

        :param assignment: assignment
        :param prob_value: probability
        :param utility_value: utility
        """
        self._matrix[assignment] = [prob_value, utility_value]

    @dispatch(Assignment, float, float)
    def increment_entry(self, assignment, prob_increments, utility_increments):
        """
        Increment the entry with new probability and utility values

        :param assignment: the assignment
        :param prob_increments: probability increment
        :param utility_increments: utility increment
        """
        old_value = self._matrix.get(assignment, [0., 0.])
        value = [old_value[0] + prob_increments, old_value[1] + utility_increments]
        self._matrix[assignment] = value

    def normalize_util(self):
        """
        Normalise the utilities with respect to the probabilities in the double
        factor.
        """
        for key in self._matrix.keys():
            value = self._matrix[key]
            if value[0] > 0. and value[1] != 0. and value[0] != 1.:
                value[1] = value[1] / value[0]

    @dispatch()
    def normalize(self):
        """
        Normalises the factor, assuming no conditional variables in the factor.
        """
        total = sum([value[0] for value in self._matrix.values()])
        for key in self._matrix.keys():
            value = self._matrix[key]
            try:
                self._matrix[key] = [value[0] / total, value[1]]
            except:
                raise ValueError()

    @dispatch(Collection)
    def normalize(self, cond_vars):
        """
        Normalises the factor, with the conditional variables as argument.

        :param cond_vars: the conditional variables
        """
        totals = dict()
        for assignment in self._matrix.keys():
            condition = assignment.get_trimmed(cond_vars)
            prob = totals.get(condition, 0.)
            totals[condition] = prob + self._matrix[assignment][0]

        for key in self._matrix.keys():
            condition = key.get_trimmed(cond_vars)
            value = self._matrix[key]
            self._matrix[key] = [value[0] / totals[condition], value[1]]

    @dispatch(Collection)
    def trim(self, head_vars):
        """
        Trims the factor to the variables provided as argument.

        :param head_vars: the variables to retain.
        """
        new_matrix = dict()

        for key in self._matrix.keys():
            key.trim(head_vars)
            new_matrix[key] = self._matrix[key]

        self._matrix = new_matrix

    def is_empty(self):
        """
        Returns true if the factor is empty, e.g. either really empty, or containing
        only empty assignments.

        :return: true if the factor is empty, false otherwise
        """
        if self._matrix is None:
            return True

        for assignment in self._matrix.keys():
            if not assignment.is_empty():
                return False

        return True

    @dispatch(Assignment)
    def get_entry(self, assignment):
        """
        Get entry
        :param assignment: assignment
        """
        try:
            return self._matrix[assignment]
        except:
            # TODO: is this possible to happen?
            for key, value in self._matrix.items():
                if key == assignment:
                    return value
            raise ValueError()

    @dispatch(Assignment)
    def get_prob_entry(self, assignment):
        """
        Returns the probability for the assignment, if it is encoded in the matrix.
        Else, returns null
        
        :param assignment: the assignment
        :return: probability of the assignment
        """
        return self._matrix[assignment][0]

    @dispatch(Assignment)
    def get_utility_entry(self, assignment):
        """
        Returns the utility for the assignment, if it is encoded in the matrix. Else,
        returns null

        :param assignment: the assignment
        :return: utility for the assignment
        """
        return self._matrix[assignment][1]

    def get_assignments(self):
        """
        Returns the matrix included in the factor

        :return: the matrix
        """
        return self._matrix.keys()

    def get_prob_table(self):
        """
        Returns the probability matrix for the factor

        :return: the probability matrix
        """
        result = dict()
        for key in self._matrix.keys():
            result[key] = self._matrix[key][0]
        return result

    def get_util_table(self):
        """
        Returns the utility matrix for the factor

        :return: the utility matrix
        """
        result = dict()
        for key in self._matrix.keys():
            result[key] = self._matrix[key][1]
        return result

    @dispatch()
    def get_values(self):
        """
        Returns the set of assignments in the factor

        :return: the set of assignments
        """
        return set(self._matrix.keys())

    @dispatch(str)
    def get_values(self, variable):
        """
        Returns the set of possible values for the given variable

        :param variable: the variable label
        :return: the set of possible values
        """
        result = set()
        for key in self._matrix.keys():
            result.add(key.get_value(variable))

        return result

    def get_variables(self):
        """
        Returns the set of variables used in the assignment. It assumes that at least
        one entry exists in the matrix. Else, returns an empty list

        :return: the set of variables
        """
        if len(self._matrix) > 0:
            return set(self._matrix.keys()).pop().get_variables()
        else:
            return set()

    @dispatch(Assignment)
    def has_assignment(self, assignment):
        """
        Returns true if the factor contains the assignment, and false otherwise

        :param assignment: the assignment
        :return: true if assignment is included, false otherwise
        """
        return assignment in self._matrix
