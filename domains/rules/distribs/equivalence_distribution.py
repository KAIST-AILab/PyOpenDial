from random import Random

from bn.distribs.distribution_builder import CategoricalTableBuilder as CategoricalTableBuilder
from bn.distribs.marginal_distribution import MarginalDistribution
from bn.distribs.prob_distribution import ProbDistribution
from bn.values.boolean_val import BooleanVal
from bn.values.string_val import StringVal
from bn.values.value import Value
from bn.values.value_factory import ValueFactory
from datastructs.assignment import Assignment
from templates.template import Template

import logging
from multipledispatch import dispatch


class EquivalenceDistribution(ProbDistribution):
    """
    Representation of an equivalence distribution (see dissertation p. 78 for details)
    with two possible values: true or false. The distribution is essentially defined as:
    P(eq=true | X, X^p) = 1 when X = X^p and != None = NONE_PROB when X = None or X^p
    = None = 0 otherwise.
    """
    none_prob = 0.02

    # logger
    log = logging.getLogger('PyOpenDial')

    def __init__(self, arg1):
        if isinstance(arg1, str):
            variable = arg1
            """
            Create a new equivalence node for the given variable.
    
            :param variable: the variable label
            """
            self._base_var = variable
            self._sampler = Random()

        else:
            raise NotImplementedError()

    @dispatch(float)
    def prune_values(self, threshold):
        """
        Does nothing
        """
        return False

    def __copy__(self):
        """
        Copies the distribution
        """
        return EquivalenceDistribution(self._base_var)

    def __str__(self):
        """
        Returns a string representation of the distribution
        """
        return "Equivalence(" + self._base_var + ", " + self._base_var + "^p)"

    @dispatch(str, str)
    def modify_variable_id(self, old_id, new_id):
        """
        Replaces occurrences of the old variable identifier oldId with the new
        identifier newId.
        """
        if self._base_var == old_id:
            self._base_var = new_id.replace("'", "")

    @dispatch(Assignment)
    def sample(self, condition):
        """
        Generates a sample from the distribution given the conditional assignment.
        """
        prob = self.get_prob(condition)

        if self._sampler.next_double() < prob:
            return ValueFactory.create(True)
        else:
            return ValueFactory.create(False)

    @dispatch()
    def get_variable(self):
        """
        Returns the identifier for the equivalence distribution

        :return: a singleton set with the equality identifier
        """
        return "=_" + self._base_var

    @dispatch()
    def get_input_variables(self):
        """
        Returns the conditional variables of the equivalence distribution. (NB: not
        sure where this implementation works in all cases?)
        """
        inputs = set()
        inputs.add(self._base_var + "^p")
        inputs.add(self._base_var + "'")
        return inputs

    @dispatch(Assignment, Value)
    def get_prob(self, condition, head):
        """
        Returns the probability of P(head | condition).

        :param condition: the conditional assignment
        :param head: the head assignment
        :return: the resulting probability
        """
        try:
            prob = self.get_prob(condition)
            if isinstance(head, BooleanVal):
                val = head.get_boolean()
                if val:
                    return prob
                else:
                    return 1. - prob
            self.log.warning(
                "cannot extract prob for P(%s|%s)" % (str(head), str(condition)))
        except Exception as e:
            self.log.warning(str(e))

        return 0.0

    @dispatch(Assignment)
    def get_prob(self, condition):
        """
        Returns the probability of eq=true given the condition

        :param condition: the conditional assignment
        :return: the probability of eq=true
        """
        predicted = None
        actual = None
        for input_var in condition.get_variables():
            if input_var == self._base_var + "^p":
                predicted = condition.get_value(input_var)
            elif input_var == self._base_var + "'":
                actual = condition.get_value(input_var)
            elif input_var == self._base_var:
                actual = condition.get_value(input_var)

        if predicted is None or actual is None:
            raise ValueError()

        if predicted == ValueFactory.none() or actual == ValueFactory.none():
            return EquivalenceDistribution.none_prob
        elif predicted == actual:
            return 1.0
        elif isinstance(predicted, StringVal) and isinstance(actual, StringVal):
            str1 = str(predicted)
            str2 = str(actual)
            if Template.create(str1).match(str2).is_matching() or Template.create(str2).match(str1).is_matching():
                return 1.0
            return 0.0
        elif len(predicted.get_sub_values()) > 0 and len(actual.get_sub_values()) > 0:
            vals0 = predicted.get_sub_values()
            vals1 = actual.get_sub_values()
            intersect = set(vals0)
            intersect.intersection_update(vals1)

            return float(len(intersect) / len(vals0))
        else:
            return 0.0

    @dispatch(Assignment)
    def get_posterior(self, condition):
        """
        Returns a new equivalence distribution with the conditional assignment as fixed input.
        """
        return MarginalDistribution(self, condition)

    @dispatch(Assignment)
    def get_prob_distrib(self, condition):
        """
        Returns the categorical table associated with the conditional assignment.

        :param condition: condition the conditional assignment
        :return: the corresponding categorical table on the true and false values
                if the table could not be extracted for the condition
        """
        positive_prob = self.get_prob(condition)
        builder = CategoricalTableBuilder(self.get_variable())
        builder.add_row(True, positive_prob)
        builder.add_row(False, 1. - positive_prob)
        return builder.build()

    @dispatch()
    def get_values(self):
        """
        Returns a set of two assignments: one with the value true, and one with the
        value false.

        :return: the set with the two possible assignments
        """
        result = set()
        result.add(ValueFactory.create(True))
        result.add(ValueFactory.create(False))
        return result
