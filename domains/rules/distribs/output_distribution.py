from bn.distribs.distribution_builder import CategoricalTableBuilder as CategoricalTableBuilder
from bn.distribs.marginal_distribution import MarginalDistribution
from bn.distribs.prob_distribution import ProbDistribution
from bn.values.value import Value
from bn.values.value_factory import ValueFactory
from datastructs.assignment import Assignment
from domains.rules.distribs.anchored_rule import AnchoredRule
from domains.rules.effects.effect import Effect
from utils.inference_utils import InferenceUtils

import logging
from multipledispatch import dispatch


class OutputDistribution(ProbDistribution):
    """
    Representation of an output distribution (see Pierre Lison's PhD thesis, page 70
    for details), which is a reflection of the combination of effects specified in the
    parent rules.
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    def __init__(self, arg1):
        if isinstance(arg1, str):
            var = arg1
            """
            Creates the output distribution for the output variable label
    
            :param var: the variable name
            """
            self._base_var = var.replace("'", "")
            self._primes = var.replace(self._base_var, "")
            self._input_rules = []

        else:
            raise NotImplementedError()

    @dispatch(AnchoredRule)
    def add_anchored_rule(self, rule):
        """
        Adds an incoming anchored rule to the output distribution.

        :param rule: the incoming rule
        """
        self._input_rules.append(rule)

    @dispatch(str, str)
    def modify_variable_id(self, old_id, new_id):
        """
        Modifies the label of the output variable.

        :param old_id: the old label
        :param new_id: the new label
        """
        if (self._base_var + self._primes) == old_id:
            self._base_var = new_id.replace("'", "")
            self._primes = new_id.replace(self._base_var, "")

    @dispatch(Assignment)
    def sample(self, condition):
        """
        Samples a particular value for the output variable.

        :param condition: the values of the parent (rule) nodes
        :return: an assignment with the output value
        """
        result = self.get_prob_distrib(condition)
        return result.sample()

    @dispatch(float)
    def prune_values(self, threshold):
        """
        Does nothing.
        """
        return False

    @dispatch(Assignment, Value)
    def get_prob(self, condition, head):
        """
        Returns the probability associated with the given conditional and head assignments.

        :param condition: the conditional assignment
        :param head: the head assignment
        :return: the resulting probability
        """
        result = self.get_prob_distrib(condition)
        return result.get_prob(head)

    @dispatch(Assignment)
    def get_prob_distrib(self, condition):
        """
        Fills the cache with the resulting table for the given condition

        :param condition: the condition for which to fill the cache
        """
        builder = CategoricalTableBuilder(self._base_var + self._primes)

        full_effects = list()
        for inputVal in condition.get_values():
            if isinstance(inputVal, Effect):
                full_effects.extend(inputVal.get_sub_effects())

        full_effect = Effect(full_effects)
        values = full_effect.get_values(self._base_var)
        if full_effect.is_non_exclusive(self._base_var):
            add_val = ValueFactory.create(list(values.keys()))
            builder.add_row(add_val, 1.0)
        elif len(values) > 0:
            total = 0.0
            for f in values.values():
                total += float(f)
            for v in values.keys():
                builder.add_row(v, values[v] / total)
        else:
            builder.add_row(ValueFactory.none(), 1.0)

        return builder.build()

    @dispatch(Assignment)
    def get_posterior(self, condition):
        """
        Returns the probability table associated with the condition

        :param condition: the conditional assignment
        :return: the resulting probability table
        """
        return MarginalDistribution(self, condition)

    @dispatch()
    def get_values(self):
        """
        Returns the possible outputs values given the input range in the parent nodes
        (probability rule nodes)

        :return: the possible values for the output
        """
        values = set()

        for rule in self._input_rules:
            for effect in rule.get_effects():
                if effect.is_non_exclusive(self._base_var):
                    return self.get_values_linearise()
                set_values = set(effect.get_values(self._base_var).keys())
                if set_values:
                    values.update(set_values)
                else:
                    values.add(ValueFactory.none())

        if len(values) == 0:
            values.add(ValueFactory.none())

        return values

    @dispatch()
    def get_variable(self):
        """
        Returns a singleton set with the label of the output

        :return: the singleton set with the output label
        """
        return self._base_var + self._primes

    @dispatch()
    def get_input_variables(self):
        """
        Returns the set of identifiers for all incoming rule nodes.
        """
        result = set()
        for rule in self._input_rules:
            result.update(rule.get_variable())
        return result

    def __copy__(self):
        """
        Returns a copy of the distribution
        """
        copy = OutputDistribution(self._base_var + self._primes)
        for rule in self._input_rules:
            copy.add_anchored_rule(rule)
        return copy

    def __str__(self):
        """
        Returns "(output)"
        """
        return "(output)"

    @dispatch()
    def get_values_linearise(self):
        """
        Calculates the possible values for the output distribution via linearisation
        (more costly operation, but necessary in case of add effects).

        :return: the set of possible output values
        """
        table = dict()
        for i in range(len(self._input_rules)):
            table[str(i)] = self._input_rules[i].get_effects()

        combinations = InferenceUtils.get_all_combinations(table)

        values = set()
        for cond in combinations:
            values.update(self.get_prob_distrib(cond).get_values())

        if len(values) == 0:
            values.add(ValueFactory.none())

        return values
