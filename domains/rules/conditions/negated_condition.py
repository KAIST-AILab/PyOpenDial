import abc

from datastructs.assignment import Assignment
from domains.rules import rule_grounding
from domains.rules.conditions.condition import Condition
from domains.rules.rule_grounding import RuleGrounding

import logging
from multipledispatch import dispatch


class NegatedCondition(Condition):
    """
    Negated condition, which is satisfied when the included condition is not.
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    # ===================================
    # CONDITION CONSTRUCTION
    # ===================================

    def __init__(self, arg1):
        if isinstance(arg1, Condition):
            condition = arg1
            """
            Creates a new negated condition with the condition provided as argument
    
            :param condition: the condition to negate
            """
            self._init_condition = condition

        else:
            raise NotImplementedError()

    # ===================================
    # GETTERS
    # ===================================

    @dispatch()
    def get_input_variables(self):
        """
        Returns the input variables for the condition (which are the same as the ones
        for the condition to negate)

        :return: the input variables
        """
        return self._init_condition.get_input_variables()

    @dispatch(Assignment)
    def get_groundings(self, param):
        g = self._init_condition.get_groundings(param)
        g = RuleGrounding() if g.is_failed() else g
        return g

    @dispatch(Assignment)
    def is_satisfied_by(self, param):
        """
        Returns true if the condition to negate is *not* satisfied, and false if it is satisfied

        :param param: the input assignment to verify
        :return: true if the included condition is false, and vice versa
        """
        return not self._init_condition.is_satisfied_by(param)

    @dispatch()
    def get_init_condition(self):
        """
        Returns the condition to negate

        :return: the condition to negate
        """
        return self._init_condition

    @dispatch()
    def get_slots(self):
        """
        Returns the list of slots in the condition
        :return: the list of slots
        """
        return self._init_condition.get_slots()

    # ===================================
    # UTILITY FUNCTIONS
    # ===================================

    def __hash__(self):
        """
        Returns the hashcode for the condition

        :return: the hashcode
        """
        return -hash(self._init_condition)

    def __str__(self):
        """
        Returns the string representation of the condition
        """
        return "!" + str(self._init_condition)

    def __eq__(self, other):
        """
        Returns true if the current instance and the object are identical, and false otherwise

        :param other: the object to compare
        :return: true if equal, false otherwise
        """
        if isinstance(other, NegatedCondition):
            return self._init_condition == other.get_init_condition()
        return False
