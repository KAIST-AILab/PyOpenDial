from collections import Collection

from bn.values.value import Value
from datastructs.assignment import Assignment

import logging
from multipledispatch import dispatch


class RuleGroundingWrapper:
    pass


class RuleGrounding(RuleGroundingWrapper):
    """
    Representation of a set of possible groundings for a rule
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    def __init__(self, arg1=None):
        if arg1 is None:
            """
            Constructs an empty set of groundings
            """
            self._groundings = set()
            self._groundings.add(Assignment())

        elif isinstance(arg1, Assignment):
            assigns = arg1
            """
            Constructs a set of groundings with the given collection of alternative
            assignments
    
            :param assigns: the assignments
            """
            self._groundings = set()
            self._groundings.add(Assignment())
            self.extend(assigns)

        else:
            raise NotImplementedError()

    @dispatch(Collection)
    def add(self, alternatives):
        """
        Adds a collection of alternative groundings. At least one must be non-failed.

        :param alternatives: the alternative groundings
        """
        found_success = False

        for g in alternatives:
            if not g.is_failed():
                self.add(g)
                found_success = True
        if not found_success:
            self._groundings.clear()

    @dispatch(RuleGroundingWrapper)
    def add(self, other):
        """
        Adds a list of alternative groundings to the existing set

        :param other: the alternative groundings
        """
        for other_assign in other._groundings:
            found = False
            for g in self._groundings:
                if g.contains(other_assign):
                    found = True
                    break
            if found:
                continue
            self._groundings.add(other_assign)
        if not self.is_empty():
            if Assignment() in self._groundings:
                self._groundings.remove(Assignment())

    @dispatch(Assignment)
    def add(self, single_assign):
        """
        Adds a single assignment to the list of alternative groundings

        :param single_assign: the assignment
        """
        if single_assign.is_empty():
            return
        for g in self._groundings:
            if g.contains(single_assign):
                return
        self._groundings.add(single_assign)
        if not self.is_empty():
            if Assignment() in self._groundings:
                self._groundings.remove(Assignment())

    @dispatch(Assignment)
    def extend(self, assign):
        """
        Extends the existing groundings with the provided assignment

        :param assign: the new assignment
        """
        if assign.is_empty():
            return
        for g in self._groundings:
            g.add_assignment(assign)

    @dispatch(RuleGroundingWrapper)
    def extend(self, other):
        """
        Extends the existing groundings with the alternative groundings

        :param other: the next groundings to extend the current ones
        """
        if other.is_failed():
            self._groundings.clear()
            return
        self.extend(other.get_alternatives())

    @dispatch(Collection)
    def extend(self, alternatives):
        """
        Extends the existing groundings with the alternative groundings

        :param alternatives: the next groundings to extend the current ones
        """
        if alternatives.is_empty():
            return
        new_groundings = set()
        for o in alternatives:
            for g in self._groundings:
                new_groundings.add(Assignment([o, g]))
                new_groundings.add(Assignment([g, o]))
        self._groundings = new_groundings

    @dispatch(str, Collection)
    def extend(self, variable, vals):
        """
        Extend a set of groundings with the alternative values for the variable

        :param variable: the variable label
        :param vals: the alternative values
        """
        new_groundings = set()
        for g in self._groundings:
            for v in vals:
                new_groundings.add(Assignment(g, variable, v))
        self._groundings = new_groundings

    @dispatch()
    def set_as_failed(self):
        """
        Sets the groundings as failed (i.e. no possible groundings for the
        underspecified variables).
        """
        self._groundings.clear()

    @dispatch()
    def get_alternatives(self):
        """
        Returns the set of possible assignments
        :return: the possible assignments
        """
        return self._groundings

    @dispatch()
    def is_failed(self):
        return not self._groundings

    @dispatch(set)
    def remove_variables(self, variables):
        """
        Removes the given variables from the assignments

        :param variables: the variable labels
        """
        for a in self._groundings:
            a.remove_all(variables)

    @dispatch(Value)
    def remove_value(self, value):
        """
        Removes the given value from the assignments

        :param value: the variable value
        """
        for a in self._groundings:
            a.remove_values(value)

    def __copy__(self):
        """
        Copies the groundings

        :return: the copy
        """
        return RuleGrounding(self._groundings)

    def __hash__(self):
        """
        Returns the hashcode of the set of alternative assignments
        """
        return hash(self._groundings)

    def __str__(self):
        """
        Returns a string representation of the set of assignments
        """
        if not self.is_failed():
            return str(self._groundings)
        else:
            return "failed"

    __repr__ = __str__

    def __eq__(self, other):
        """
        Returns true if o is a rule grounding with the same assignments, false otherwise
        """
        if isinstance(other, RuleGrounding):
            return self._groundings == other._groundings
        return False

    @dispatch()
    def is_empty(self):
        """
        Returns true if the groundings are empty, false otherwise

        :return: true if empty, false otherwise
        """
        if len(self._groundings) == 0:
            return True

        if len(self._groundings) == 1:
            item = self._groundings.pop()
            self._groundings.add(item)
            if item.is_empty():
                return True

        return False
