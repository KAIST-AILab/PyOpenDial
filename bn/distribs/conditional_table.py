import logging
from copy import copy

from multipledispatch import dispatch

from bn.distribs.independent_distribution import IndependentDistribution
from bn.distribs.prob_distribution import ProbDistribution
from bn.distribs.single_value_distribution import SingleValueDistribution
from bn.values.value import Value
from bn.values.value_factory import ValueFactory
from datastructs.assignment import Assignment


class ConditionalTable(ProbDistribution):
    """
    Conditional probability distribution represented as a probability table. The table
    expresses a generic distribution of type P(X|Y1...Yn), where X is called the
    "head" random variable, and Y1...Yn the conditional random variables.

    Constructing a conditional table should be done via its Builder class:
    builder = ConditionalTableBuilder("variable name")
    builder.addRow(...)
    table = builder.build()

    This class represent a generic conditional distribution in which the distribution
    for the head variable X can be represented using arbitrary distributions of type
    IndependentProbDistribution.
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    # ===================================
    # TABLE CONSTRUCTION
    # ===================================

    def __init__(self, arg1=None, arg2=None):
        if isinstance(arg1, str) and arg2 is None:
            head_var = arg1
            """
            Constructs a new probability table, with no values

            :param head_var: the name of the random variable
            """
            self._table = dict()
            self._head_var = head_var
            self._conditional_vars = set()
        elif isinstance(arg1, str) and isinstance(arg2, dict):
            head_var, distribs = arg1, arg2
            """
            Constructs a new probability table, with the values in distribs

            :param head_var: the name of the random variable
            :param distribs: the distribs (one for each conditional assignment)
            """
            self._table = dict()
            self._head_var = head_var
            self._conditional_vars = set()
            for condition in distribs.keys():
                self.add_distrib(condition, distribs[condition])
        else:
            raise NotImplementedError()

    @dispatch(str, str)
    def modify_variable_id(self, old_id, new_id):
        """
        Modifies the distribution table by replace the old variable identifier by the new one

        :param old_id: the old variable label
        :param new_id: the new variable label
        """
        for condition in self._table.keys():
            self._table[condition].modify_variable_id(old_id, new_id)
            if condition.contains_var(old_id):
                # TODO: check refactor > something is weird
                distrib = self._table.pop(condition)
                value = condition.remove_pair(old_id)
                condition.add_pair(new_id, value)
                self._table[condition] = distrib

        if old_id in self._conditional_vars:
            self._conditional_vars.remove(old_id)
            self._conditional_vars.add(new_id)

        if self._head_var == old_id:
            self._head_var = new_id

    @dispatch(Assignment, IndependentDistribution)
    def add_distrib(self, condition, distrib):
        """
        Adds a new continuous probability distribution associated with the given conditional assignment

        :param condition: the conditional assignment
        :param distrib: the distribution (in a continuous, function-based representation)
                        @ if distrib relates to a different random variable
        """
        self._table[condition] = distrib
        if distrib.get_variable() != self._head_var:
            raise ValueError("Variable is %s, not %s" % (self._head_var, distrib.get_variable()))

        self._conditional_vars.update(condition.get_variables())

    # TODO: check bug > something is weird.
    @dispatch(float)
    def prune_values(self, threshold):
        changed = False

        for condition in self._table.keys():
            changed = changed or self._table[condition].prune_values(threshold)

        return changed

    # ===================================
    # GETTERS
    # ===================================

    @dispatch()
    def get_variable(self):
        """
        Returns the name of the random variable

        :return: the (head) random variable
        """
        return self._head_var

    @dispatch(Assignment)
    def sample(self, condition):
        """
        Sample a head assignment from the distribution P(head|condition), given the
        condition. If no assignment can be sampled (due to e.g. an ill-formed
        distribution), returns an empty assignment.

        :param condition: the condition
        :return: the sampled assignment the condition
        """
        if condition.size() != len(self._conditional_vars):
            condition = condition.get_trimmed(self._conditional_vars)

        subdistrib = self._table[condition]
        if subdistrib is not None:
            return subdistrib.sample()

        # TODO: check refactor > raise exception?
        return ValueFactory.none()

    # TODO: check refactor > raise exception?
    @dispatch(Assignment, Value)
    def get_prob(self, condition, value):
        """
        Returns the probability of the head assignment given the conditional
        assignment. The method assumes that the posterior distribution has a discrete
        form.

        :param condition: the conditional assignment
        :param value: the head assignment
        :return: the resulting probability
        """
        if condition.size() > len(self._conditional_vars):
            condition = condition.get_trimmed(self._conditional_vars)

        if condition in self._table:
            return self._table[condition].get_prob(value)
        elif condition.is_default():
            total_prob = 0.
            for a in self._table.keys():
                total_prob += self._table[a].get_prob(value)
            return total_prob
        else:
            # self.log.warning("could not find the corresponding condition for %s)" % condition)
            return 0.

    @dispatch(Assignment)
    def get_prob_distrib(self, condition):
        """
        Returns the (unconditional) probability distribution P(X) given the conditional assignment.

        :param condition: the conditional assignment
        :return: the corresponding probability distribution
        """
        if condition in self._table:
            return self._table[condition]
        else:
            # TODO: check refactor > raise exception?
            return SingleValueDistribution(self._head_var, ValueFactory.none())

    @dispatch(Assignment)
    def get_posterior(self, condition):
        """
        Returns the posterior distribution obtained by integrating the (possibly
        partial) conditional assignment.

        :param condition: the assignment on a subset of the conditional variables
        :return: the resulting posterior distribution.
        """
        if condition in self._table:
            return self._table[condition]

        new_distrib = ConditionalTable(self._head_var)
        for a in self._table.keys():
            if a.consistent_with(condition):
                remaining = a.get_pruned(condition.get_variables())
                if remaining not in new_distrib._table:
                    new_distrib.add_distrib(remaining, self._table[a])
                else:
                    # TODO: check refactor > raise exception?
                    self.log.warning("inconsistent results for partial posterior")
                    pass

        return new_distrib

    @dispatch()
    def get_values(self):
        """
        Returns all possible values specified in the table. The input values are here
        ignored (for efficiency reasons), so the method simply extracts all possible
        head rows in the table.

        :return: the possible values for the head variables.
        """
        head_rows = set()
        for condition in self._table.keys():
            head_rows.update(self._table[condition].get_values())
        return head_rows

    @dispatch()
    def get_conditions(self):
        """
        Returns the set of possible conditional assignments in the table.

        :return: the set of conditional assignments
        """
        return self._table.keys()

    @dispatch()
    def get_input_variables(self):
        """
        Returns the conditional variables of the table

        :return: the set of conditional variables
        """
        return self._conditional_vars

    # ===================================
    # UTILITIES
    # ===================================

    def __hash__(self):
        """
        Returns the hashcode for the table.
        """
        return hash(frozenset(self._table.items()))

    def __copy__(self):
        """
        Returns a copy of the probability table

        :return: the copy
        """
        new_table = ConditionalTable(self._head_var)
        for condition in self._table.keys():
            try:
                new_table.add_distrib(condition, copy(self._table[condition]))
            except Exception as e:
                self.log.warning("Copy error: %s" % e)

        return new_table

    def __str__(self):
        """
        Returns a pretty print of the distribution

        :return: the pretty print
        """
        result = []
        for condition in self._table.keys():
            distrib = self._table[condition]
            for value in distrib.get_values():
                prob = distrib.get_prob(value)
                if condition.size() > 0:
                    result.append('P(%s=%s|%s):=%f' % (self._head_var, str(value), str(condition), prob))
                else:
                    result.append('P(%s=%s):=%f' % (self._head_var, str(value), prob))

        return '\n'.join(result)

    def __eq__(self, other):
        """
        Returns true if the object o is a conditional distribution with the same content
        """
        if not isinstance(other, ConditionalTable):
            return False

        if self._table.keys() != other._table.keys():
            return False

        for o_assignment in other._table.keys():
            s_distrib = self._table[o_assignment]
            o_distrib = other._table[o_assignment]

            if s_distrib != o_distrib:
                return False

        return True

