from enum import Enum, auto

from bn.values.array_val import ArrayVal
from bn.values.none_val import NoneVal
from bn.values.set_val import SetVal
from bn.values.value import Value
from bn.values.value_factory import ValueFactory
from datastructs.assignment import Assignment
from domains.rules.conditions.condition import Condition
from domains.rules.rule_grounding import RuleGrounding
from templates.template import Template

import logging
from multipledispatch import dispatch


class Relation(Enum):
    EQUAL = auto()
    UNEQUAL = auto()
    CONTAINS = auto()
    NOT_CONTAINS = auto()
    GREATER_THAN = auto()
    LOWER_THAN = auto()
    IN = auto()
    NOT_IN = auto()
    LENGTH = auto()


class BasicConditionWrapper(Condition):
    pass


class BasicCondition(BasicConditionWrapper):
    """
    Basic condition between a variable and a value
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    # ===================================
    # CONDITION CONSTRUCTION
    # ===================================

    def __init__(self, arg1=None, arg2=None, arg3=None):
        if isinstance(arg1, str) and isinstance(arg2, str) and isinstance(arg3, Relation):
            variable, value, relation = arg1, arg2, arg3
            """
            Creates a new basic condition, given a variable label, an expected value, and
            a relation to hold between the variable and its value
    
            :param variable: the variable
            :param value: the value
            :param relation: the relation to hold
            """
            self._variable = Template.create(variable)
            self._template_value = Template.create(value)
            self._ground_value = None if self._template_value.is_under_specified() else ValueFactory.create(value)
            self._relation = relation

        elif isinstance(arg1, str) and isinstance(arg2, Value) and isinstance(arg3, Relation):
            variable, value, relation = arg1, arg2, arg3
            """
            Creates a new basic condition, given a variable label, an expected value, and
            a relation to hold between the variable and its value
    
            :param variable: the variable
            :param value: the value
            :param relation: the relation to hold
            """
            self._variable = Template.create(variable)
            self._template_value = Template.create(str(value))
            self._ground_value = value
            self._relation = relation

        elif isinstance(arg1, BasicConditionWrapper) and isinstance(arg2, Assignment) and arg3 is None:
            condition, grounding = arg1, arg2
            """
            Creates a new basic condition that represented the grounding of the provided
            condition together with the value assignment
    
            :param condition: the condition (with free variables)
            :param grounding: the grounding assignment
            """
            self._variable = condition._variable

            if self._variable.is_under_specified():
                self._variable = Template.create(self._variable.fill_slots(grounding))

            self._relation = condition._relation
            self._template_value = condition._template_value
            self._ground_value = condition._ground_value

            if len(self._template_value.get_slots()) > 0:
                self._template_value = Template.create(self._template_value.fill_slots(grounding))

                if not self._template_value.is_under_specified():
                    self._ground_value = ValueFactory.create(str(self._template_value))

        else:
            raise NotImplementedError()

    # ===================================
    # GETTERS
    # ===================================

    @dispatch()
    def get_relation(self):
        """
        Returns the relation in place for the condition

        :return: the relation
        """
        return self._relation

    @dispatch()
    def get_variable(self):
        """
        Returns the variable label for the basic condition

        :return: the variable label
        """
        return self._variable

    @dispatch()
    def get_input_variables(self):
        """
        Returns the input variables for the condition (the main variable itself, plus
        optional slots in the value to fill)

        :return: the input variables
        """
        return [self._variable]

    @dispatch()
    def get_slots(self):
        """
        Returns the slots in the variable and value template
        """
        slots = set()
        slots.update(self._variable.get_slots())
        slots.update(self._template_value.get_slots())
        return slots

    @dispatch(Assignment)
    def is_satisfied_by(self, param):
        """
        Returns true if the condition is satisfied by the value assignment provided as
        argument, and false otherwise
        This method uses an external ConditionCheck object to ease the process.

        :param param: the actual assignment of values
        :return: true if the condition is satisfied, false otherwise
        """
        if not self._variable.is_filled_by(param) or not self._template_value.is_filled_by(param):
            return False

        grounded = BasicCondition(self, param)
        actual_value = param.get_value(str(grounded._variable))
        return grounded.is_satisfied(actual_value)

    @dispatch(Value)
    def is_satisfied(self, actual_value):
        """
        Returns true if the relation is satisfied between the actual and expected values.

        :param actual_value: the actual value
        :return: true if satisfied, false otherwise
        """
        if self._ground_value is not None:
            if self._relation == Relation.EQUAL:
                return actual_value == self._ground_value
            elif self._relation == Relation.UNEQUAL:
                return not actual_value == self._ground_value
            elif self._relation == Relation.GREATER_THAN:
                return not actual_value.__lt__(self._ground_value)
            elif self._relation == Relation.LOWER_THAN:
                return actual_value.__lt__(self._ground_value)
            elif self._relation == Relation.CONTAINS:
                return self._ground_value in actual_value
            elif self._relation == Relation.NOT_CONTAINS:
                return self._ground_value not in actual_value
            elif self._relation == Relation.LENGTH:
                return len(actual_value) == len(self._ground_value)
            elif self._relation == Relation.IN:
                return actual_value in self._ground_value
            elif self._relation == Relation.NOT_IN:
                return actual_value not in self._ground_value

        else:
            if self._relation == Relation.EQUAL:
                return self._template_value.match(str(actual_value)).is_matching()
            elif self._relation == Relation.UNEQUAL:
                return not self._template_value.match(str(actual_value)).is_matching()
            elif self._relation == Relation.CONTAINS:
                return self._template_value.partial_match(str(actual_value)).is_matching()
            elif self._relation == Relation.NOT_CONTAINS:
                return not self._template_value.partial_match(str(actual_value)).is_matching()
            elif self._relation == Relation.LENGTH:
                return self._template_value.match("" + str(len(str(actual_value)))).is_matching()

        return False

    @dispatch(Assignment)
    def get_groundings(self, param):
        """
        Returns the set of possible groundings for the given input assignment

        :param param: the input assignment
        :return: the set of possible (alternative) groundings for the condition
        """
        ground_cond = BasicCondition(self, param)
        groundings = RuleGrounding()

        if len(ground_cond._variable.get_slots()) > 0:
            for inputVar in param.get_variables():
                m = ground_cond._variable.match(inputVar)
                if m.is_matching():
                    new_input = Assignment([param, m])
                    spec_grounds = self.get_groundings(new_input)
                    spec_grounds.extend(m)
                    groundings.add(spec_grounds)
            return groundings

        filled_var = str(ground_cond._variable)
        if len(ground_cond._template_value.get_slots()) > 0:
            actual_value = param.get_value(str(ground_cond._variable))
            groundings = ground_cond.get_groundings(actual_value)
            groundings.remove_variables(param.get_variables())
            groundings.remove_value(ValueFactory.none())

        elif self._relation == Relation.IN and not param.contains_var(filled_var):
            values_coll = ground_cond._ground_value.get_sub_values()
            groundings.extend(filled_var, values_coll)

        elif not self.is_satisfied_by(param):
            groundings.set_as_failed()

        return groundings

    @dispatch(Value)
    def get_groundings(self, param):
        """
        Tries to match the template with the actual value, and returns the associated groundings

        :param param: the actual filled value
        :return: the resulting groundings
        """
        grounding = RuleGrounding()

        if self._relation == Relation.EQUAL or self._relation == Relation.UNEQUAL:
            m = self._template_value.match(str(param))
            if m.is_matching():
                grounding.add(m)

        elif self._relation == Relation.CONTAINS and not (isinstance(param, NoneVal) or isinstance(param, SetVal) or isinstance(param, ArrayVal)):
            m2 = self._template_value.find(str(param), 100)
            for match in m2:
                grounding.add(match)

        elif self._relation == Relation.CONTAINS and param.get_sub_values():
            for sub_val in param.get_sub_values():
                m2 = self._template_value.match(str(sub_val))
                if m2.is_matching():
                    grounding.add(m2)

        if (not grounding) and self._relation != Relation.UNEQUAL:
            grounding.set_as_failed()

        return grounding

    # ===================================
    # UTILITY FUNCTIONS
    # ===================================

    def __str__(self):
        """
        Returns a string representation of the condition
        """
        if self._relation == Relation.EQUAL:
            return str(self._variable) + "=" + str(self._template_value)
        elif self._relation == Relation.UNEQUAL:
            return str(self._variable) + "!=" + str(self._template_value)
        elif self._relation == Relation.GREATER_THAN:
            return str(self._variable) + ">" + str(self._template_value)
        elif self._relation == Relation.LOWER_THAN:
            return str(self._variable) + "<" + str(self._template_value)
        elif self._relation == Relation.CONTAINS:
            return str(self._variable) + " contains " + str(self._template_value)
        elif self._relation == Relation.NOT_CONTAINS:
            return str(self._variable) + " does not contains " + str(self._template_value)
        elif self._relation == Relation.LENGTH:
            return "length(" + str(self._variable) + ")=" + str(self._template_value)
        elif self._relation == Relation.IN:
            return str(self._variable) + " in " + str(self._template_value)
        elif self._relation == Relation.NOT_IN:
            return str(self._variable) + " not in " + str(self._template_value)
        else:
            return ""

    def __hash__(self):
        """
        Returns the hashcode for the condition

        :return: the hashcode
        """
        return hash(self._variable) + hash(self._template_value) - 3 * hash(self._relation)

    def __eq__(self, other):
        """
        Returns true if the given object is a condition identical to the current
        instance, and false otherwise

        :param other: the object to compare
        :return: true if the condition are equals, and false otherwise
        """
        if isinstance(other, BasicCondition):
            if self._variable == other.get_variable():
                if self._template_value == other._template_value and self._relation == other.get_relation():
                    return True
        return False
