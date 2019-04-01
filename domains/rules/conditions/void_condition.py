import abc

from datastructs.assignment import Assignment
from domains.rules.conditions.condition import Condition
from domains.rules.rule_grounding import RuleGrounding

import logging
from multipledispatch import dispatch


class VoidCondition(Condition):
    """
    Representation of a void condition, which is always true.
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    def __init__(self):
        pass

    @dispatch()
    def get_input_variables(self):
        """
        Return an empty list

        :return: an empty list
        """
        return list()

    @dispatch(Assignment)
    def is_satisfied_by(self, param):
        """
        Returns true (condition is always trivially satisfied)

        :param param: the input assignment (ignored)
        :return: true
        """
        return True

    @dispatch(Assignment)
    def get_groundings(self, param):
        """
        Returns an empty set of groundings
        """
        return RuleGrounding()

    @dispatch()
    def get_slots(self):
        """
        Returns an empty list

        :return: an empty list
        """
        return set()

    def __str__(self):
        """
        Returns the string "true" indicating that the condition is always trivially true
        """
        return "true"

    def __hash__(self):
        """
        Returns a constant representing the hashcode for the void condition

        :return: 36
        """
        return 36

    def __eq__(self, other):
        """
        Returns true if o is also a void condition

        :param other: the object to compare
        :return: true if other is also a void condition
        """
        return isinstance(other, VoidCondition)
