import logging
import random
from collections import Collection

import numpy as np
from multipledispatch import dispatch

from bn.distribs.continuous_distribution import ContinuousDistribution
from bn.distribs.density_functions.kernel_density_function import KernelDensityFunction
from bn.distribs.distribution_builder import CategoricalTableBuilder as CategoricalTableBuilder, \
    ConditionalTableBuilder as ConditionalTableBuilder, MultivariateTableBuilder as MultivariateTableBuilder
from bn.distribs.multivariate_distribution import MultivariateDistribution
from bn.values.array_val import ArrayVal
from bn.values.double_val import DoubleVal
from datastructs.assignment import Assignment


class EmpiricalDistribution(MultivariateDistribution):
    """
    Distribution defined "empirically" in terms of a set of samples on a collection of
    random variables. This distribution can then be explicitly converted into a table
    or a continuous distribution (depending on the variable type).
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    # ===================================
    # CONSTRUCTION METHODS
    # ===================================

    def __init__(self, arg1=None):
        if arg1 is None:
            """
            Constructs an empirical distribution with an empty set of samples
            """
            self._samples = []
            self._variables = set()
            self._discrete_cache = None
            self._continuous_cache = None
        elif isinstance(arg1, Collection):
            samples = arg1
            """
            Constructs a new empirical distribution with a set of samples samples

            :param samples: the samples to add
            """
            self._samples = samples
            self._variables = set()
            self._discrete_cache = None
            self._continuous_cache = None

            for sample in self._samples:
                self._variables.update(sample.get_variables())
        else:
            raise NotImplementedError()

    @dispatch(Assignment)
    def add_sample(self, sample):
        """
        Adds a new sample to the distribution

        :param sample: the sample to add
        """
        self._samples.append(sample)
        self._discrete_cache = None
        self._continuous_cache = None
        self._variables.difference_update(sample.get_variables())

    @dispatch(str)
    def remove_variable(self, variable_id):
        """
        Removes a particular variable from the sampled assignments

        :param variable_id: the id of the variable to remove
        """

        self._variables.remove(variable_id)
        self._discrete_cache = None
        self._continuous_cache = None
        for assignment in self._samples:
            assignment.remove_pair(variable_id)

    # ===================================
    # GETTERS
    # ===================================

    @dispatch()
    def sample(self):
        """
        Samples from the distribution. In this case, simply selects one arbitrary
        sample out of the set defining the distribution

        :return: the selected sample
        """
        if len(self._samples) > 0:
            return self._samples[random.randint(0, len(self._samples) - 1)]
        else:
            self.log.warning("distribution has no samples")

        raise ValueError()

    @dispatch()
    def get_variables(self):
        """
        Returns the head variables for the distribution.
        """
        return set(self._variables)

    @dispatch()
    def get_samples(self):
        """
        Returns the collection of samples.

        :return: the collection of samples
        """
        return self._samples

    def __len__(self):
        """
        Returns the number of samples.

        :return: the number of samples
        """
        return len(self._samples)

    @dispatch()
    def get_values(self):
        """
        Returns the possible values for the variables of the distribution.

        :return: the possible values for the variables
        """
        return set(self._samples)

    @dispatch(Assignment)
    def get_prob(self, head):
        """
        Returns the probability of a particular assignment
        """
        return self.to_discrete().get_prob(head)

    @dispatch()
    def get_best(self):
        """
        Returns the value that occurs most often in the set of samples
        """
        return self.to_discrete().get_best()

    # ===================================
    # CONVERSION METHODS
    # ===================================

    @dispatch()
    def to_discrete(self):
        """
        Returns a discrete representation of the empirical distribution.
        """
        if self._discrete_cache is None:
            builder = MultivariateTableBuilder()
            prob = 1. / len(self._samples)

            for assignment in self._samples:
                trimmed_assignment = assignment.get_trimmed(self._variables)
                builder.increment_row(trimmed_assignment, prob)

            self._discrete_cache = builder.build()

        return self._discrete_cache

    @dispatch()
    def to_continuous(self):
        """
        Returns a continuous representation of the distribution (if there is no
        conditional variables in the distribution).

        :return: the corresponding continuous distribution. content is discrete.
        """
        if self._continuous_cache is None:
            if len(self._variables) != 1:
                raise ValueError("cannot convert distribution to continuous for P(%s)" % self._variables)

            self._continuous_cache = self.create_continuous(list(self._variables)[0])

        return self._continuous_cache

    @dispatch(str)
    def get_marginal(self, variable):
        """
        Returns an independent probability distribution on a single random variable
        based on the samples. This distribution may be a categorical table or a
        continuous distribution.

        :return: the probability distribution resulting from the marginalisation.
        """
        if not self.sample().get_trimmed([variable]).contain_continuous_values():
            return self.create_discrete(variable)

        # TODO: check bug or refactor. Why 5?
        if len(set(self._samples)) < 5:
            return self.create_discrete(variable)

        return self.create_continuous(variable)

    @dispatch(str, set)
    def get_marginal(self, variable, condition_variables):
        """
        Returns a distribution P(var|condvars) based on the samples. If the
        conditional variables are empty, returns an independent probability
        distribution.

        :param variable: the head variable
        :param condition_variables: the conditional variables
        :return: the resulting probability distribution be generated.
        """
        if len(condition_variables) == 0:
            return self.get_marginal(variable)

        builder = ConditionalTableBuilder(variable)
        prob = 1. / len(self._samples)

        for assignment in self._samples:
            condition = assignment.get_trimmed(condition_variables)
            value = assignment.get_value(variable)
            builder.increment_row(condition, value, prob)

        builder.normalize()
        return builder.build()

    @dispatch(str)
    def create_discrete(self, head_variable):
        """
        Creates a categorical table with the defined head variable given the samples

        :param head_variable: the variable for which to create the distribution
        :return: the resulting table
        """
        builder = CategoricalTableBuilder(head_variable)
        prob = 1. / len(self._samples)

        for assignment in self._samples:
            value = assignment.get_value(head_variable)
            builder.increment_row(value, prob)

        return builder.build()

    @dispatch(str)
    def create_continuous(self, head_variable):
        """
        Creates a continuous with the defined head variable given the samples

        :param head_variable: the variable for which to create the distribution
        :return: the resulting continuous distribution
        """
        values = []
        for assignment in self._samples:
            value = assignment.get_value(head_variable)
            if isinstance(value, ArrayVal):
                values.append(value.get_array())
            elif isinstance(value, DoubleVal):
                values.append([value.get_double()])
        values = np.array(values)

        return ContinuousDistribution(head_variable, KernelDensityFunction(values))

    # ===================================
    # UTILITY METHODS
    # ===================================

    @dispatch(float)
    def prune_values(self, threshold):
        """
        Prunes all samples that contain a value whose relative frequency is below the
        threshold specified as argument. DoubleVal and ArrayVal are ignored.

        :param threshold: the frequency threshold
        """

        freq = dict()
        for assignment in self._samples:
            for variable in assignment.get_variables():
                value = assignment.get_value(variable)
                value_freq = freq[variable]
                if value_freq is None:
                    value_freq = dict()
                    freq[variable] = value_freq

                if value not in value_freq:
                    value_freq[value] = 1
                else:
                    value_freq[value] += 1

        changed = False
        min_number = int(len(self._samples) * threshold)
        for assignment in self._samples:
            for variable in assignment.get_variables():
                if freq[variable][assignment.get_value(variable)] < min_number:
                    # TODO: check bug > is this the right implementation of pruning?
                    self._samples.remove(assignment)
                    changed = True

        self._discrete_cache = None
        self._continuous_cache = None
        return changed

    @dispatch(str, str)
    def modify_variable_id(self, old_id, new_id):
        """
        Replace a variable label by a new one

        :param old_id: the old variable label
        :param new_id: the new variable label
        """
        if old_id in self._variables:
            self._variables.remove(old_id)
            self._variables.add(new_id)

        for assignment in self._samples:
            if assignment.contains_var(old_id):
                value = assignment.remove_pair(old_id)
                assignment.add_pair(new_id, value)

        if self._discrete_cache is not None:
            self._discrete_cache.modify_variable_id(old_id, new_id)

        if self._continuous_cache is not None:
            self._continuous_cache.modify_variable_id(old_id, new_id)

    def __copy__(self):
        """
        Returns a copy of the distribution

        :return: the copy
        """
        return EmpiricalDistribution(self._samples)

    def __str__(self):
        """
        Returns a pretty print representation of the distribution: here, tries to
        convert it to a discrete distribution, and displays its content.

        :return: the pretty print
        """
        if self._is_continuous():
            try:
                return str(self.to_continuous())
            except Exception as e:
                self.log.debug("could not convert distribution to a continuous format: %s" % e)

        return str(self.to_discrete())

    @dispatch()
    def _is_continuous(self):
        for variable in self.get_variables():
            assignment = self._samples.get(0)
            if assignment.contains_var(variable) and assignment.contains_continuous_values():
                if len(self.get_variables()) == 1:
                    return True
                else:
                    return False

        return False
