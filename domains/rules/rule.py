from enum import Enum, auto
from random import Random

from datastructs.assignment import Assignment
from datastructs.math_expression import MathExpression
from domains.rules.conditions.condition import Condition
from domains.rules.conditions.void_condition import VoidCondition
from domains.rules.effects.effect import Effect
from domains.rules.parameters.complex_parameter import ComplexParameter
from domains.rules.parameters.fixed_parameter import FixedParameter
from domains.rules.parameters.parameter import Parameter
from domains.rules.rule_grounding import RuleGrounding
from templates.template import Template

import logging
from multipledispatch import dispatch

dispatch_namespace = dict()


class RuleType(Enum):
    PROB = auto()
    UTIL = auto()


class RuleOutputWrapper:
    pass


class RuleOutput(RuleOutputWrapper):
    """
    Representation of a rule output, consisting of a set of alternative (mutually
    exclusive) effects, each being associated with a particular probability or utility
    parameter.
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    def __init__(self, rule_type):
        if isinstance(rule_type, RuleType):
            """
            Creates a new output, with a void condition and an empty list of effects
    
            :param rule_type: the type of the rule
            """
            self._rule_type = rule_type
            self._effects = dict()

        else:
            raise NotImplementedError()

    @dispatch(Effect, float)
    def add_effect(self, effect, param):
        """
        Adds an new effect and its associated probability/utility to the output'

        :param effect: the effect
        :param param: the effect's probability or utility
        """
        self.add_effect(effect, FixedParameter(param))

    @dispatch(Effect, Parameter)
    def add_effect(self, effect, param):
        """
        Adds an new effect and its associated probability/utility to the output'

        :param effect: the effect
        :param param: the parameter for the effect's probability or utility
        """
        self._effects[effect] = param

    @dispatch(Effect)
    def remove_effect(self, effect):
        """
        Removes an effect from the rule output

        :param effect: the effect to remove
        """
        del self._effects[effect]

    @dispatch(Assignment)
    def ground(self, grounding):
        """
        Returns a grounded version of the rule output, based on the grounding assignment.

        :param grounding: the grounding associated with the filled values
        :return: the grounded copy of the output.
        """
        ground_case = RuleOutput(self._rule_type)

        for effect in self._effects.keys():
            grounded_effect = effect.ground(grounding)

            if len(grounded_effect.get_sub_effects()) > 0 or len(effect.get_sub_effects()) == 0:
                a_param = self._effects[effect]

                if isinstance(a_param, ComplexParameter):
                    a_param = a_param.ground(grounding)

                ground_case.add_effect(grounded_effect, a_param)

        if self._rule_type == RuleType.PROB:
            ground_case.prune_effects()
            ground_case.add_void_effect()

        return ground_case

    @dispatch(RuleOutputWrapper)
    def add_output(self, new_case):
        """
        Adds a rule output to the current one. The result is a joint probability
        distribution in the output of a probability rule, and an addition of utility
        tables in the case of a utility rule.

        :param new_case: the new rule case to add
        """
        if self.is_void():
            self._effects = new_case._effects

        elif new_case.is_void() or hash(frozenset(self._effects.items())) == hash(frozenset(new_case._effects.items())):
            return

        elif self._rule_type == RuleType.PROB:
            new_output = dict()

            for effect_1 in self._effects.keys():
                param_1 = self._effects[effect_1]

                for effect_2 in new_case.get_effects():
                    param_2 = new_case.get_parameter(effect_2)
                    new_effect = Effect([effect_1, effect_2])
                    new_param = RuleOutput.merge_parameters(param_1, param_2, '*')

                    if new_effect in new_output:
                        new_param = RuleOutput.merge_parameters(new_output[new_effect], new_param, '+')

                    new_output[new_effect] = new_param

            self._effects = new_output
            new_case.prune_effects()

        elif self._rule_type == RuleType.UTIL:
            for effect in new_case.get_effects():
                # TODO: Original code seems strange. a_param is not needed here.
                # a_param = new_case.get_parameter(effect)
                # if effect in self.effects:
                #    a_param = RuleOutput.merge_parameters(self.effects[effect], a_param, '+')
                self._effects[effect] = new_case.get_parameter(effect)

    @dispatch()
    def get_effects(self):
        """
        Returns all the effects specified in the case.

        :return: the set of effects
        """
        return set(self._effects.keys())

    @dispatch(Effect)
    def get_parameter(self, effect):
        """
        Returns the parameter associated with the effect. If the effect is not part of
        the case, returns null.

        :param effect: the effect
        :return: the parameter associated with the effect
        """
        # TODO: 나중엔 'return self._effects[effect]' 로 바꿉시다.
        for (key, value) in self._effects.items():
            if key == effect:
                return value
        raise ValueError()

    @dispatch()
    def is_void(self):
        """
        Returns true if the case is void (empty or with a single void effect)

        :return: true if void, false otherwise
        """
        return (not self._effects) or (self._rule_type == RuleType.PROB and len(self._effects) == 1 and Effect() in self._effects)

    @dispatch()
    def get_output_variables(self):
        """
        Returns the set of output variables for the case, as defined in the effects
        associated with the case condition. The output variables are appended with a '
        suffix.

        :return: the set of output variables defined in the case's effects
        """
        total_variables = set()

        for effect in self._effects.keys():
            sub_variables = effect.get_output_variables()

            for variable in sub_variables:
                total_variables.add(variable + "'")

        return total_variables

    @dispatch()
    def get_pairs(self):
        """
        Returns the set of (effect,parameters) pairs in the rule case

        :return: the set of associated pairs
        """
        return set(self._effects.items())

    @dispatch()
    def get_parameters(self):
        """
        Returns the collection of parameters used in the output

        :return: the parameters
        """
        return self._effects.values()

    def __str__(self):
        """
        Returns a string representation of the rule case.
        """
        result = ""

        for effect in self._effects.keys():
            result += str(effect)
            result += " [" + str(self._effects[effect]) + "]"
            result += ","

        if len(self._effects) > 0:
            result = result[0:-1]

        return result

    def __hash__(self):
        """
        Returns the hashcode for the case

        :return: the hashcode
        """
        return -2 * hash(self._effects)

    def __eq__(self, other):
        """
        Returns true if the object is a identical rule case, and false otherwise.
        """
        return isinstance(other, RuleOutput) and self._effects == other.effects

    @dispatch()
    def prune_effects(self):
        """
        Prunes all effects whose parameter is lower than the provided threshold. This
        only works for fixed parameters.
        """
        for effect in list(self._effects.keys()):
            a_param = self._effects[effect]

            from modules.state_pruner import StatePruner
            if isinstance(a_param, FixedParameter) and a_param.get_value() < StatePruner.value_pruning_threshold:
                del self._effects[effect]

    @dispatch()
    def add_void_effect(self):
        """
        Adds a void effect to the rule if the fixed mass is lower than 1.0 and a void
        effect is not already defined.
        """
        # case 1: if there are no effects, insert a void one with prob.1
        if len(self._effects) == 0:
            self.add_effect(Effect(), FixedParameter(1.0))
            return

        fixed_mass = 0
        for effect in self._effects.keys():
            # case 2: if there is already a void effect, do nothing
            if len(effect) == 0:
                return

            # sum up the fixed probability mass
            a_param = self._effects[effect]
            if isinstance(a_param, FixedParameter):
                fixed_mass += a_param.get_value()

        # case 3: if the fixed probability mass is = 1, do nothing
        if fixed_mass > 0.99:
            return

        # case 4: if the fixed probability mass is < 1, fill the remaining mass
        elif fixed_mass > 0.0:
            self.add_effect(Effect(), FixedParameter(1 - fixed_mass))

        # case 5: in case the rule output is structured via single or complex
        # parameters p1, p2,... pn, create a new complex effect = 1 - (p1+p2+...pn)
        # that fill the remaining probability mass
        else:
            params = []
            for value in self._effects.values():
                params.append(MathExpression(value.get_expression()))
            one = MathExpression("1")
            negation = one.combine('-', params)
            self.add_effect(Effect(), ComplexParameter(negation))

    @staticmethod
    @dispatch(Parameter, Parameter, str, namespace=dispatch_namespace)
    def merge_parameters(p1, p2, operator):
        """
        Merges the two parameters and returns the merged parameter

        :param p1: the first parameter
        :param p2: the second parameter
        :param operator: the operator, such as +, * or -
        :return: the resulting parameter
        """
        if isinstance(p1, FixedParameter) and isinstance(p2, FixedParameter):
            v1 = p1.get_value()
            v2 = p2.get_value()
            if operator == '+':
                return FixedParameter(v1 + v2)
            elif operator == '*':
                return FixedParameter(v1 * v2)
            elif operator == '-':
                return FixedParameter(v1 - v2)
            else:
                raise ValueError()

        exp1 = p1.get_expression()
        exp2 = p2.get_expression()
        return ComplexParameter(exp1.combine(operator, [exp2]))


class Rule:
    """
    Generic representation of a probabilistic rule, with an identifier and an ordered
    list of cases. The rule can be either a probability or a utility rule.
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    def __init__(self, arg1, arg2):
        if isinstance(arg1, str) and isinstance(arg2, RuleType):
            id, rule_type = arg1, arg2
            """
            Creates a new rule, with the given identifier and type, and an empty list of cases
    
            :param id: the identifier
            :param rule_type: the rule type
            """
            self._id = id
            self._rule_type = rule_type
            self._cases = []

        else:
            raise NotImplementedError()

    @dispatch(Condition, RuleOutput)
    def add_case(self, condition, output):
        """
        Adds a new case to the abstract rule

        :param condition: the condition
        :param output: the corresponding output
        """
        if len(self._cases) > 0 and isinstance(self._cases[-1]._condition, VoidCondition):
            self.log.warning("unreachable case for rule %s (previous case trivially true)" % self._id)

        if self._rule_type == RuleType.PROB:
            total_mass = 0
            for p in output.get_parameters():
                if isinstance(p, FixedParameter):
                    pv = p.get_value()
                    if pv < 0.0:
                        raise ValueError()
                    total_mass += pv
            # TODO: eps 확인 필요.
            eps = 1e-8
            if total_mass > 1. + eps:
                raise ValueError()

        self._cases.append(RuleCase(condition, output))

    @dispatch()
    def get_rule_id(self):
        """
        Returns the rule identifier

        :return: the rule identifier
        """
        return self._id

    @dispatch()
    def get_input_variables(self):
        """
        Returns the input variables (possibly underspecified, with slots to fill) for the rule

        :return: the set of labels for the input variables
        """
        input_vars = set()
        for c in self._cases:
            input_vars.update(c.get_input_variables())
        return input_vars

    @dispatch(Assignment)
    def get_output(self, assignment):
        """
        Returns the first rule output whose condition matches the input assignment
        provided as argument. The output contains the grounded list of effects
        associated with the satisfied condition.

        :param assignment: the input assignment
        :return: the matched rule output
        """
        output = RuleOutput(self._rule_type)
        groundings = self.get_groundings(assignment)
        for g in groundings.get_alternatives():
            full = Assignment([assignment, g]) if not g.is_empty() else assignment

            match = None
            for c in self._cases:
                if c._condition.is_satisfied_by(full):
                    match = c._output
                    break
            if match is None:
                match = RuleOutput(self._rule_type)

            match = match.ground(full)
            output.add_output(match)
        return output

    @dispatch()
    def get_rule_type(self):
        """
        Returns the rule type

        :return: the rule type
        """
        return self._rule_type

    @dispatch()
    def get_parameter_ids(self):
        """
        Returns the set of all parameter identifiers employed in the rule

        :return: the set of parameter identifiers
        """
        params = set()
        for c in self._cases:
            for e in c.get_effects():
                params.update(c.output.get_parameter(e).get_variables())
        return params

    @dispatch()
    def get_effects(self):
        """
        Returns the set of all possible effects in the rule.

        :return: the set of all possible effects
        """
        effects = set()
        for c in self._cases:
            effects.update(c.get_effects())
        return effects

    @dispatch(Assignment)
    def get_groundings(self, assignment):
        """
        Returns the set of groundings that can be derived from the rule and the
        specific input assignment.

        :param assignment: the input assignment
        :return: the possible groundings for the rule
        """
        groundings = RuleGrounding()
        for c in self._cases:
            new_grounding = c.get_groundings(assignment, self._rule_type)
            groundings.add(new_grounding)
        return groundings

    def __str__(self):
        """
        Returns a string representation for the rule
        """
        rule_cases_str = '\telse'.join([str(rule_case) + '\n' for rule_case in self._cases])
        return self._id + ': ' + rule_cases_str

    def __hash__(self):
        """
        Returns the hashcode for the rule

        :return: the hashcode
        """
        return hash(self.__class__) - hash(self._id) + hash(self._cases)

    def __eq__(self, other):
        """
        Returns true if o is a rule that has the same identifier, rule type and list
        of cases than the current rule.

        :param other: the object to compare
        :return: true if the object is an identical rule, false otherwise.
        """
        if isinstance(other, Rule):
            return self._id == other.get_rule_id() and self._rule_type == other.get_rule_type() \
                   and self._cases == other._cases
        return False


class RuleCase:
    """
    Representation of a rule case, i.e. a condition associated with a rule output
    """

    def __init__(self, arg1, arg2):
        if isinstance(arg1, Condition) and isinstance(arg2, RuleOutput):
            condition, output = arg1, arg2
            """
            Creates a new rule case with a condition and an output
    
            :param condition: the condition
            :param output: the associated output
            """
            self._condition = condition
            self._output = output

        else:
            raise NotImplementedError()

    @dispatch()
    def get_effects(self):
        """
        Returns the set of effects for the rule case

        :return: the set of effects
        """
        return self._output.get_effects()

    @dispatch(Assignment, RuleType)
    def get_groundings(self, assign, rule_type):
        """
        Returns the groundings associated with the rule case, given the input assignment

        :param assign: the input assignment
        :param rule_type: the resulting groundings
        """
        groundings = RuleGrounding()
        new_groundings = self._condition.get_groundings(assign)
        groundings.add(new_groundings)

        if rule_type == RuleType.UTIL:
            action_vars = assign.contains_vars(self._output.get_output_variables())
            for effect in self.get_effects():
                if action_vars:
                    condition = effect.convert_to_condition()
                    effect_grounding = condition.get_groundings(assign)
                    groundings.add(effect_grounding)
                else:
                    slots = effect.get_value_slots()
                    for variable in assign.get_variables():
                        if variable in slots:
                            slots.remove(variable)
                    for variable in self._condition.get_input_variables():
                        if str(variable) in slots:
                            slots.remove(str(variable))
                    groundings.add(Assignment.create_one_value(slots, ""))

        for effect in self.get_effects():
            for random_to_generate in effect.get_randoms_to_generate():
                groundings.extend(Assignment(random_to_generate, Random().next_int(99999)))

        return groundings

    @dispatch()
    def get_input_variables(self):
        """
        Returns the input variables associated with the case

        :return: the set of input variables
        """
        input_vars = set(self._condition.get_input_variables())

        for slot in self._condition.get_slots():
            template = Template.create(slot)
            input_vars.add(template)

        for effect in self.get_effects():
            for input_variable in effect.get_value_slots():
                input_vars.add(Template.create(input_variable))

        return input_vars

    def __str__(self):
        """
        Returns a string representation of the rule case.
        """
        res = ""
        if not isinstance(self._condition, VoidCondition):
            res += "if (" + str(self._condition) + ") then "
        else:
            res += " "
        res += str(self._output)
        return res

    __repr__ = __str__
