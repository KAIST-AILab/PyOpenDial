import sys
from collections import Collection

from bn.values.set_val import SetVal
from bn.values.value import Value
from bn.values.value_factory import ValueFactory
from datastructs.assignment import Assignment
from domains.rules.conditions.complex_condition import ComplexCondition, BinaryOperator
from domains.rules.conditions.void_condition import VoidCondition
from domains.rules.effects.basic_effect import BasicEffect
from domains.rules.effects.template_effect import TemplateEffect
from templates.template import Template

import logging
from multipledispatch import dispatch

dispatch_namespace = dict()


class Effect(Value):
    """
    A complex effect, represented as a combination of elementary sub-effects connected
    via an implicit AND relation.
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    # ===================================
    # EFFECT CONSTRUCTION
    # ===================================

    def __init__(self, arg1=None):
        if arg1 is None:
            """
            Creates a new complex effect with no effect
            """
            self._equivalent_condition = None

            self._sub_effects = list()
            self._fully_grounded = True
            self._randoms_to_generate = set()
            self._value_table = dict()

        elif isinstance(arg1, BasicEffect):
            effect = arg1
            """
            Creates a new complex effect with a single effect
    
            :param effect: the effect to include
            """
            self._equivalent_condition = None

            self._sub_effects = [effect]
            self._fully_grounded = not effect.contains_slots()
            self._value_table = dict()
            self._randoms_to_generate = set()

            if isinstance(effect, TemplateEffect):
                all_slots = effect.get_all_slots()
                for s in all_slots:
                    if s.startswith("random"):
                        self._randoms_to_generate.add(s)

        elif isinstance(arg1, Collection) or isinstance(arg1, list):
            effects = arg1
            """
            Creates a new complex effect with a collection of existing effects
    
            :param effects: the effects to include
            """
            if len(effects) == 0 or isinstance(effects[0], BasicEffect):
                self._sub_effects = list(effects)
                self._equivalent_condition = None

                self._fully_grounded = True
                for effect in effects:
                    if effect.contains_slots():
                        self._fully_grounded = False
                        break
                self._value_table = dict()
                self._randoms_to_generate = set()

                for effect in effects:
                    if isinstance(effect, TemplateEffect):
                        all_slots = effect.get_all_slots()
                        for s in all_slots:
                            if s.startswith("random"):
                                self._randoms_to_generate.add(s)

            elif isinstance(effects[0], Effect):
                basic_effects = []
                for effect in effects:
                    basic_effects += effect.get_sub_effects()
                self.__init__(basic_effects)

        else:
            raise NotImplementedError()

    # ===================================
    # GETTERS
    # ===================================

    @dispatch()
    def get_sub_effects(self):
        """
        Returns all the sub-effect included in the complex effect

        :return: the collection of sub-effects
        """
        return self._sub_effects

    @dispatch(Assignment)
    def ground(self, grounding):
        """
        Grounds the effect with the given assignment.

        :param grounding: the assignment containing the filled values
        :return: the resulting grounded effect
        """
        if self._fully_grounded:
            return self

        grounded = list()
        for effect in self._sub_effects:
            sub_grounds = effect.ground(grounding)
            if not sub_grounds.contains_slots():
                grounded.append(sub_grounds)

        return Effect(grounded)

    @dispatch(Value)
    def concatenate(self, value):
        if isinstance(value, Effect):
            effects = list()
            effects.extend(value.get_sub_effects())
            return Effect(effects)
        else:
            raise ValueError()

    @dispatch()
    def get_value_slots(self):
        """
        Returns the additional input variables for the complex effect

        :return: the set of labels for the additional input variables
        """
        effects = list()
        for effect in self._sub_effects:
            if isinstance(effect, TemplateEffect):
                effects.extend(effect.get_value_template().get_slots())

        return set(effects)

    @dispatch()
    def get_output_variables(self):
        """
        Returns the output variables for the complex effect (including all the output
        variables for the sub-effects)

        :return: the set of all output variables
        """
        variables = set()
        for sub_effect in self._sub_effects:
            variables.add(sub_effect.get_variable())
        return variables

    @dispatch(str)
    def get_values(self, variable):
        """
        Returns the set of values specified in the effect for the given variable.
        If several effects are defined with distinct priorities, only the effect with
        highest priority is retained.

        :param variable: the variable
        :return: the values specified in the effect
        """
        if variable not in self._value_table:
            self._value_table[variable] = self.create_table(variable)
        return self._value_table[variable]

    @dispatch(str)
    def is_non_exclusive(self, variable):
        """
        Returns true if all of the included effects for the variable are not mutually
        exclusive (allowing multiple values).

        :param variable: the variable to check
        :return: true if the effects are not mutually exclusive, else false
        """
        non_exclusive = False
        for sub_effect in self._sub_effects:
            if sub_effect.get_variable() == variable:
                if not sub_effect.is_exclusive():
                    non_exclusive = True
                elif len(sub_effect.get_value()) > 0 and not sub_effect.is_negated():
                    return False
        return non_exclusive

    @dispatch()
    def convert_to_condition(self):
        """
        Converts the effect into a condition.

        :return: the corresponding condition
        """
        if self._equivalent_condition is None:
            conditions = list()
            for sub_effect in self.get_sub_effects():
                conditions.append(sub_effect.convert_to_condition())
            if len(conditions) == 0:
                self._equivalent_condition = VoidCondition()
            elif len(conditions) == 1:
                self._equivalent_condition = conditions[0]
            elif len(self.get_output_variables()) == 1:
                self._equivalent_condition = ComplexCondition(conditions, BinaryOperator.OR)
            else:
                self._equivalent_condition = ComplexCondition(conditions, BinaryOperator.AND)
        return self._equivalent_condition

    def __len__(self):
        """
        Returns the number of basic effects

        :return: the number of effects
        """
        return len(self._sub_effects)

    @dispatch()
    def get_assignment(self):
        """
        Returns the effect as an assignment of values. The variable labels are ended
        by a prime character.

        :return: the assignment of new values to the variables
        """
        assignment = Assignment()
        for effect in self._sub_effects:
            if not effect._negated:
                assignment.add_pair(effect.get_variable() + "'", effect.get_value())
            else:
                assignment.add_pair(effect.get_variable() + "'", ValueFactory.none())
        return assignment

    @dispatch()
    def get_randoms_to_generate(self):
        """
        Returns the slots of the form {randomX}, which are to be replaced by random integers.

        :return: the set of slots to replace by random integers
        """
        return self._randoms_to_generate

    # ===================================
    # UTILITY FUNCTIONS
    # ===================================

    def __hash__(self):
        """
        Returns the hashcode for the complex effect

        :return: the hashcode
        """
        return hash(tuple(self._sub_effects))

    def __eq__(self, other):
        """
        Returns true if the object is a complex effect with an identical content

        :param other: the object to compare
        :return: true if the objects are identical, false otherwise
        """
        return hash(self) == hash(other)

    def __str__(self):
        """
        Returns a string representation for the effect
        """
        s = ""
        for e in self._sub_effects:
            s += str(e) + " ^ "
        return s[0:-3] if len(self._sub_effects) > 0 else "Void"

    __repr__ = __str__

    def __copy__(self):
        """
        Returns a copy of the effect

        :return: the copy
        """
        effects = list()
        for effect in self._sub_effects:
            effects.append(effect)
        return Effect(effects)

    @dispatch(Value)
    def __contains__(self, sub_value):
        """
        Returns false.
        """
        return False

    @dispatch(Value)
    def compare_to(self, other):
        """
        Compares the effect with another value (based on their hashcode).
        """
        return hash(self) - hash(other)

    @dispatch()
    def get_sub_values(self):
        """
        Returns an empty list
        """
        return list()

    @staticmethod
    @dispatch(str, namespace=dispatch_namespace)
    def parse_effect(str_val):
        """
        Parses the string representing the effect, and returns the corresponding effect.

        :param str_val: the string representing the effect
        :return: the corresponding effect
        """
        if " ^ " in str_val:
            effects = list()
            for split in str_val.split(" ^ "):
                sub_output = Effect.parse_effect(split)
                effects += sub_output.get_sub_effects()
            return Effect(effects)

        else:
            if "Void" in str_val:
                return Effect(list())

            var = ""
            val = ""

            exclusive = True
            negated = False

            if ":=" in str_val:
                var = str_val.split(":=")[0]
                val = str_val.split(":=")[1]
                val = "None" if "{}" in val else val
            elif "!=" in str_val:
                var = str_val.split("!=")[0]
                val = str_val.split("!=")[1]
                negated = True
            elif "+=" in str_val:
                var = str_val.split("+=")[0]
                val = str_val.split("+=")[1]
                exclusive = False

            tvar = Template.create(var)
            tval = Template.create(val)
            if tvar.is_under_specified() or tval.is_under_specified():
                return Effect(TemplateEffect(tvar, tval, 1, exclusive, negated))
            else:
                return Effect(BasicEffect(var, ValueFactory.create(val), 1, exclusive, negated))

    @dispatch(str)
    def create_table(self, variable):
        """
        Extracts the values (along with their weight) specified in the effect.

        :param variable: variable the variable
        :return: the corresponding values
        """
        values = dict()

        max_priority = sys.maxsize
        to_remove = set()
        to_remove.add(ValueFactory.none())

        for effect in self._sub_effects:
            if effect.get_variable() == variable:
                if effect._priority < max_priority:
                    max_priority = effect._priority
                if effect._negated:
                    to_remove.add(effect.get_value())

        for effect in self._sub_effects:
            effect_variable = effect.get_variable()
            if effect_variable == variable:
                effect_value = effect.get_value()
                if effect._priority > max_priority or effect._negated or effect_value in to_remove:
                    continue
                if len(to_remove) > 1 and isinstance(effect_value, SetVal):
                    sub_values = effect_value.get_sub_values()
                    for r in to_remove:
                        if r in sub_values:
                            sub_values.remove(r)

                    effect_value = ValueFactory.create(sub_values)

                if effect_value not in values:
                    values[effect_value] = effect._weight
                else:
                    values[effect_value] += effect._weight

        return values
