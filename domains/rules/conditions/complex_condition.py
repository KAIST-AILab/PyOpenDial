from enum import Enum, auto

from datastructs.assignment import Assignment
from domains.rules.rule_grounding import RuleGrounding
from domains.rules.conditions.condition import Condition

import logging
from multipledispatch import dispatch


class BinaryOperator(Enum):
    AND = auto()
    OR = auto()


class ComplexCondition(Condition):
    """
    Complex condition made up of a collection of sub-conditions connected with a
    logical operator (AND, OR).
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    # ===================================
    # CONDITION CONSTRUCTION
    # ===================================

    def __init__(self, arg1, arg2):
        if isinstance(arg1, list) and isinstance(arg2, BinaryOperator):
            sub_conditions, operator = arg1, arg2
            """
            Creates a new complex condition with a list of subconditions
    
            :param sub_conditions: the subconditions
            :param operator: the binary operator to employ between the conditions
            """
            if operator is None:
                operator = BinaryOperator.AND

            self._sub_conditions = sub_conditions
            self._operator = operator

        else:
            raise NotImplementedError()

    # ===================================
    # GETTERS
    # ===================================

    @dispatch()
    def get_operator(self):
        """
        Returns the logical operator for the complex condition

        :return: the operator
        """
        return self._operator

    @dispatch()
    def get_input_variables(self):
        """
        Returns the set of input variables for the complex condition

        :return: the set of input variables
        """
        variables = []
        for cond in self._sub_conditions:
            variables += cond.get_input_variables()
        return variables

    @dispatch()
    def get_conditions(self):
        """
        Returns the subconditions in the complex condition.

        :return: the subconditions
        """
        return self._sub_conditions

    @dispatch()
    def get_slots(self):
        """
        Returns the list of all slots used in the conditions

        :return: the list of all slots
        """
        slots = set()
        for cond in self._sub_conditions:
            slots.update(cond.get_slots())
        return slots

    @dispatch(Assignment)
    def is_satisfied_by(self, param):
        """
        Returns true if the complex condition is satisfied by the input assignment,
        and false otherwise.
        If the logical operator is AND, all the subconditions must be satisfied. If
        the operator is OR, at least one must be satisfied.

        :param param: the input assignment
        :return: true if the conditions are satisfied, false otherwise
        """
        for cond in self._sub_conditions:
            if self._operator == BinaryOperator.AND and not cond.is_satisfied_by(param):
                return False
            elif self._operator == BinaryOperator.OR and cond.is_satisfied_by(param):
                return True
        return self._operator == BinaryOperator.AND

    @dispatch(Assignment)
    def get_groundings(self, param):
        """
        Returns the groundings for the complex condition (which is the union of the
        groundings for all basic conditions).

        :return: the full set of groundings
        """
        groundings = RuleGrounding()

        if self._operator == BinaryOperator.AND:
            for cond in self._sub_conditions:
                new_grounding = RuleGrounding()
                found_grounding = False
                for g in groundings.get_alternatives():
                    g2 = param if g.is_empty() else Assignment([param, g])
                    ground = cond.get_groundings(g2)
                    found_grounding = found_grounding or not ground.is_failed()
                    ground.extend(g)
                    new_grounding.add(ground)
                if not found_grounding:
                    new_grounding.set_as_failed()
                    return new_grounding
                groundings = new_grounding
        elif self._operator == BinaryOperator.OR:
            alternatives = []
            for cond in self._sub_conditions:
                alternatives.append(cond.get_groundings(param))
            groundings.add(alternatives)

        return groundings

    # ===================================
    # UTILITY FUNCTIONS
    # ===================================

    def __str__(self):
        """
        Returns a string representation of the complex condition
        """
        s = ""
        for cond in self._sub_conditions:
            s += str(cond)
            op = self._operator
            if op == BinaryOperator.AND:
                s += " ^ "
            elif op == BinaryOperator.OR:
                s += " v "
        return s[0:-3]

    def __hash__(self):
        """
        Returns the hashcode for the condition

        :return: the hashcode
        """
        return hash(self._sub_conditions) - hash(self._operator)

    def __eq__(self, other):
        """
        Returns true if the complex conditions are equal, false otherwise

        :param other: the object to compare with current instance
        :return: true if the conditions are equal, false otherwise
        """
        return hash(self) == hash(other)
