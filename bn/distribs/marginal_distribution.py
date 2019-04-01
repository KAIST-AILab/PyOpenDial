import logging
import threading
from copy import copy

from multipledispatch import dispatch

from bn.distribs.categorical_table import CategoricalTable
from bn.distribs.distribution_builder import CategoricalTableBuilder as CategoricalTableBuilder
from bn.distribs.multivariate_distribution import MultivariateDistribution
from bn.distribs.multivariate_table import MultivariateTable
from bn.distribs.prob_distribution import ProbDistribution
from bn.values.value import Value
from datastructs.assignment import Assignment


class MarginalDistribution(ProbDistribution):
    """
    Representation of a probability distribution P(X|Y1,...Yn) by way of two
    distributions:
    - a distribution P(X|Y1,...Yn, Z1,...Zm)
    - a distribution P(Z1,...Zm)

    The probability P(X|Y1,...Yn) can be straightforwardly calculated by marginalising
    out the variables Z1,...Zm.
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    def __init__(self, arg1=None, arg2=None):
        if isinstance(arg1, ProbDistribution) and isinstance(arg2, MultivariateDistribution):
            cond_distrib, uncond_distrib = arg1, arg2
            """
            Creates a new marginal distribution given the two component distributions.

            :param cond_distrib: the distribution P(X|Y1,...Yn, Z1,...Zm)
            :param uncond_distrib: the distributionP(Z1,...Zm)
            """
            self._cond_distrib = cond_distrib
            self._uncond_distrib = uncond_distrib
            self._init_lock()
        elif isinstance(arg1, ProbDistribution) and isinstance(arg2, Assignment):
            cond_distrib, assign = arg1, arg2
            """
            Creates a new marginal distribution given the first distributions and the
            value assignment (corresponding to a distribution P(Z1,...Zm) where the
            provided assignment has a probability 1.0).

            :param cond_distrib: the distribution P(X|Y1,...Yn, Z1,...Zm)
            :param assign:  the assignment of values for Z1,...Zm
            """
            self._cond_distrib = cond_distrib
            self._uncond_distrib = MultivariateTable(assign)
            self._init_lock()
        elif isinstance(arg1, ProbDistribution) and isinstance(arg2, CategoricalTable):
            cond_distrib, uncond_distrib = arg1, arg2
            """
            Creates a new marginal distribution given the first distribution and the
            categorical table P(Z).

            :param cond_distrib: the distribution P(X|Y1,...Yn, Z)
            :param uncond_distrib:  the distribution P(Z).
            """

            self._cond_distrib = cond_distrib
            self._uncond_distrib = MultivariateTable(uncond_distrib)
            self._init_lock()
        else:
            raise NotImplementedError()

    def _init_lock(self):
        # TODO: need refactoring (decorator?)
        self._locks = {
            'prune_values': threading.RLock(),
            'modify_variable_id': threading.RLock(),
        }

    @dispatch()
    def get_variable(self):
        """
        Returns the variable X.
        """
        return self._cond_distrib.get_variable()

    @dispatch()
    def get_input_variables(self):
        """
        Returns the conditional variables Y1,...Yn for the distribution.

        :return: the set of conditional variables
        """
        inputs = set(self._cond_distrib.get_input_variables())
        inputs.difference_update(self._uncond_distrib.get_variables())
        return inputs

    @dispatch()
    def get_conditional_distrib(self):
        """
        Returns the conditional distribution P(X|Y1,...Yn,Z1,...Zm)

        :return: the conditional distribution
        """
        return self._cond_distrib

    @dispatch(Assignment, Value)
    def get_prob(self, condition, head):
        """
        Returns the probability P(X=head|condition)

        :param condition: the conditional assignment for Y1,...Yn
        :param head:
        :return:
        """
        total_prob = 0.
        for assignment in self._uncond_distrib.get_values():
            augmented_condition = Assignment([condition, assignment])
            total_prob += self._cond_distrib.get_prob(augmented_condition, head)

        return total_prob

    @dispatch(Assignment)
    def sample(self, condition):
        """
        Returns a sample value for the variable X given the conditional assignment

        :param condition: the conditional assignment for Y1,...Yn
        :return: the sampled value for X
        """
        return self._cond_distrib.sample(Assignment([condition, self._uncond_distrib.sample()]))

    @dispatch(Assignment)
    def get_prob_distrib(self, condition):
        """
        Returns the categorical table P(X) given the conditional assignment for the
        variables Y1,...Yn.

        :param condition: the conditional assignment for Y1,...Yn
        :return: the categorical table for the random variable X
        """
        builder = CategoricalTableBuilder(self._cond_distrib.get_variable())
        for assignment in self._uncond_distrib.get_values():
            prob = self._uncond_distrib.get_prob(assignment)
            augmented_condition = Assignment([condition, assignment])
            categorical_table = self._cond_distrib.get_prob_distrib(augmented_condition).to_discrete()
            for value in categorical_table.get_values():
                builder.increment_row(value, prob * categorical_table.get_prob(value))

        return builder.build()

    @dispatch()
    def get_values(self):
        """
        Returns the possible values for X.

        :return: the set of possible values
        """
        return self._cond_distrib.get_values()

    @dispatch(Assignment)
    def get_posterior(self, condition):
        """
        Returns the posterior distribution given the conditional assignment.

        :param condition: condition a conditional assignment on a subset of variables from Y1,...Yn
        :return: the resulting posterior distribution.
        """
        table = self._uncond_distrib.to_discrete()
        table.extend_rows(condition)

        return MarginalDistribution(self._cond_distrib, table)

    @dispatch(float)
    def prune_values(self, threshold):
        """
        Prune the values below a certain threshold.

        :param threshold: the threshold to apply
        """
        with self._locks['prune_values']:
            changed = self._cond_distrib.prune_values(threshold)
            changed = changed or self._uncond_distrib.prune_values(threshold)
            return changed

    def __copy__(self):
        """
        Copies the marginal distribution.
        """
        return MarginalDistribution(copy(self._cond_distrib), copy(self._uncond_distrib))

    @dispatch(str, str)
    def modify_variable_id(self, old_id, new_id):
        """
        Modifies the variable identifier in the two distributions.
        """
        with self._locks['modify_variable_id']:
            self._cond_distrib.modify_variable_id(old_id, new_id)
            self._uncond_distrib.modify_variable_id(old_id, new_id)

    def __str__(self):
        """
        Returns a text representation of the marginal distribution.
        """
        return 'Marginal distribution with %s and %s' % (str(self._cond_distrib), str(self._uncond_distrib))
