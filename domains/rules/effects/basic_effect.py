from bn.values.value import Value
from bn.values.value_factory import ValueFactory
from datastructs.assignment import Assignment
from domains.rules.conditions.basic_condition import BasicCondition, Relation

import logging
from multipledispatch import dispatch


class BasicEffect:
    """
    Representation of a basic effect of a rule. A basic effect is formally defined as a triple with:
    - a variable label;
    - one of four basic operations on the variable SET, DISCARD, ADD;
    - a variable value;
    This class represented a usual, fully grounded effect. For effects including
    underspecified entities, use TemplateEffect.
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    # ===================================
    # EFFECT CONSTRUCTION
    # ===================================

    def __init__(self, arg1=None, arg2=None, arg3=None, arg4=None, arg5=None):
        if isinstance(arg1, str) and isinstance(arg2, str) and arg3 is None and arg4 is None and arg5 is None:
            variable, value = arg1, arg2
            """
            Constructs a new basic effect, with a variable label and value.
    
            :param variable: variable label (raw string)
            :param value: variable value (raw string)
            """
            self.__init__(variable, ValueFactory.create(value), 1, True, False)

        elif isinstance(arg1, str) and isinstance(arg2, Value) and isinstance(arg3, int) and isinstance(arg4, bool) and isinstance(arg5, bool):
            variable, value, priority, exclusive, negated = arg1, arg2, arg3, arg4, arg5
            """
            Constructs a new basic effect, with a variable label, value and other
            arguments. The argument "add" specifies whether the effect is mutually
            exclusive with other effects. The argument "negated" specifies whether the
            effect includes a negation.
    
            :param variable: variable label (raw string)
            :param value: variable value
            :param priority: the priority level (default is 1)
            :param exclusive: whether distinct values are mutually exclusive or not
            :param negated: whether to negate the effect or not
            """
            self._variable_label = variable
            self._variable_value = value
            self._priority = priority
            self._exclusive = exclusive
            self._negated = negated
            self._weight = 1.0

        else:
            raise NotImplementedError()

    # ===================================
    # GETTERS
    # ===================================

    @dispatch()
    def get_variable(self):
        """
        Returns the variable label for the basic effect

        :return: the variable label
        """
        return self._variable_label

    @dispatch()
    def get_value(self):
        """
        Returns the variable value for the basic effect. If the variable value is
        underspecified, returns the value None.

        :return: the varuabke value
        """
        return self._variable_value

    @dispatch()
    def convert_to_condition(self):
        """
        Converts the effect into an equivalent condition

        :return: the corresponding condition
        """
        r = Relation.UNEQUAL if self._negated else Relation.EQUAL
        return BasicCondition(self._variable_label + "'", self._variable_value, r)

    @dispatch()
    def contains_slots(self):
        """
        Returns false.

        :return: false
        """
        return False

    @dispatch(Assignment)
    def ground(self, grounding):
        """
        Returns itself.

        :param grounding: the grounding assignment
        :return: itself
        """
        return self

    @dispatch()
    def get_priority(self):
        """
        Returns the rule priority

        :return: the priority level
        """
        return self._priority

    @dispatch()
    def get_weight(self):
        """
        Returns the effect weight

        :return: the weight
        """
        return self._weight

    @dispatch()
    def is_exclusive(self):
        """
        Returns true if the effect allows only one distinct value for the variable
        (default case) and false otherwise

        :return: true if the effect allows only one distinct value, else false.
        """
        return self._exclusive

    @dispatch()
    def is_negated(self):
        """
        Returns true is the effect is negated and false otherwise.

        :return: whether the effect is negated
        """
        return self._negated

    # ===================================
    # UTILITY METHODS
    # ===================================

    def __str__(self):
        """
        Returns the string representation of the basic effect
        """
        s = self._variable_label
        if self._negated:
            s += "!="
        elif not self._exclusive:
            s += "+="
        else:
            s += ":="
        s += str(self._variable_value)
        return s

    def __hash__(self):
        """
        Returns the hashcode for the basic effect

        :return: the hashcode
        """
        return (-2 if self._negated else 1) * hash(self._variable_label) ^ hash(self._exclusive) ^ self._priority ^ hash(self._variable_value)

    def __eq__(self, other):
        """
        Returns true if the object o is a basic effect that is identical to the
        current instance, and false otherwise.

        :param other: the object to compare
        :return: true if the objects are identical, false otherwise
        """
        if isinstance(other, BasicEffect):
            if self._variable_label is not other.get_variable():
                return False
            elif self.get_value() is not other.get_value():
                return False
            elif self._exclusive is not other._exclusive:
                return False
            elif self._negated is not other.is_negated():
                return False
            elif self._priority is not other._priority:
                return False
            return True
        return False

    def __copy__(self):
        """
        Returns a copy of the effect.

        :return: the copy
        """
        return BasicEffect(self._variable_label, self._variable_value, self._priority, self._exclusive, self._negated)

    @dispatch(int)
    def change_priority(self, priority):
        """
        Changes the priority of the basic effects

        :param priority: the new priority
        """
        self._priority = priority
