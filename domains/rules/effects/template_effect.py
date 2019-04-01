from bn.values.value_factory import ValueFactory
from datastructs.assignment import Assignment
from domains.rules.effects.basic_effect import BasicEffect
from domains.rules.conditions.basic_condition import BasicCondition, Relation
from templates.template import Template

import logging
from multipledispatch import dispatch


class TemplateEffect(BasicEffect):
    """
    Representation of a basic effect of a rule. A basic effect is formally defined as
    a triple with:
    - a (possibly underspecified) variable label;
    - one of four basic operations on the variable SET, DISCARD, ADD;
    - a (possibly underspecified) variable value;
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    # ===================================
    # EFFECT CONSTRUCTION
    # ===================================

    def __init__(self, arg1=None, arg2=None, arg3=1, arg4=True, arg5=False):
        if isinstance(arg1, Template) and isinstance(arg2, Template) and isinstance(arg3, int) and isinstance(arg4, bool) and isinstance(arg5, bool):
            variable, value, priority, exclusive, negated = arg1, arg2, arg3, arg4, arg5
            """
            Constructs a new effect, with a variable label, value, and other arguments.
            The argument "add" specifies whether the effect is mutually exclusive with
            other effects. The argument "negated" specifies whether the effect includes
            a negation.
    
            :param variable: variable label
            :param value: variable value
            :param priority:the priority level (default is 1)
            :param exclusive: whether distinct values are mutually exclusive or not
            :param negated: whether to negate the effect or not.
            """
            super(TemplateEffect, self).__init__(str(variable),
                                                 ValueFactory.none() if value.is_under_specified() else ValueFactory.create(str(value)),
                                                 priority, exclusive, negated)

            self._label_template = variable
            self._value_template = value

        else:
            raise NotImplementedError()

    @dispatch(Assignment)
    def ground(self, grounding):
        """
        Ground the slots in the effect variable and value (given the assignment) and
        returns the resulting effect.

        :param grounding: the grounding
        :return: the grounded effect
        """
        new_t = Template.create(self._label_template.fill_slots(grounding))
        new_v = Template.create(self._value_template.fill_slots(grounding))
        if new_t.is_under_specified() or new_v.is_under_specified():
            return TemplateEffect(new_t, new_v, self._priority, self._exclusive, self._negated)
        else:
            return BasicEffect(str(new_t), ValueFactory.create(str(new_v)), self._priority, self._exclusive, self._negated)

    # ===================================
    # GETTERS
    # ===================================

    @dispatch()
    def contains_slots(self):
        """
        Returns true if the effect contains slots to fill, and false otherwise
        """
        return len(self._label_template.get_slots()) > 0 or len(self._value_template.get_slots()) > 0

    @dispatch()
    def convert_to_condition(self):
        """
        Converts the effect into an equivalent condition

        :return: the corresponding condition
        """
        r = Relation.UNEQUAL if self._negated else Relation.EQUAL
        return BasicCondition(str(self._label_template) + "'", str(self._value_template), r)

    @dispatch()
    def get_variable_template(self):
        """
        Returns the template representation of the variable label

        :return: the variable template
        """
        return self._label_template

    @dispatch()
    def get_value_template(self):
        """
        Returns the template representation of the variable value

        :return: the value template
        """
        return self._value_template

    @dispatch()
    def get_all_slots(self):
        """
        Returns all slots in the template effects

        :return: the set of all slots
        """
        all_slots = set()
        all_slots.update(self._label_template.get_slots())
        all_slots.update(self._value_template.get_slots())
        return all_slots

    # ===================================
    # UTILITY METHODS
    # ===================================

    def __str__(self):
        """
        Returns the string representation of the basic effect
        """
        s = str(self._label_template)
        if self._negated:
            s += "!="
        elif not self._exclusive:
            s += "+="
        else:
            s += ":="
        s += str(self._value_template)
        return s

    def __hash__(self):
        """
        Returns the hashcode for the basic effect

        :return: the hashcode
        """
        return (-2 if self._negated else 1) * hash(self._label_template) ^ hash(self._exclusive) ^ self._priority ^ hash(self._value_template)

    def __eq__(self, other):
        """
        Returns true if the object o is a basic effect that is identical to the
        current instance, and false otherwise.

        :param other: the object to compare
        :return: true if the objects are identical, false otherwise
        """
        if isinstance(other, TemplateEffect):
            if self._label_template is not other._label_template:
                return False
            elif self._value_template is not other._value_template:
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
        return TemplateEffect(self._label_template, self._value_template, self._priority, self._exclusive, self._negated)
