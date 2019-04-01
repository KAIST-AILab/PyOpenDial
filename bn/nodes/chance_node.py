from multipledispatch import dispatch
from copy import copy
import logging

from bn.distribs.continuous_distribution import ContinuousDistribution
from bn.distribs.independent_distribution import IndependentDistribution
from bn.distribs.prob_distribution import ProbDistribution
from bn.distribs.single_value_distribution import SingleValueDistribution
from bn.nodes.action_node import ActionNode
from bn.nodes.b_node import BNode
from bn.values.value import Value
from datastructs.assignment import Assignment
from settings import Settings


class ChanceNode(BNode):
    """
    Representation of a chance node (sometimes also called belief node), which is a
    random variable associated with a specific probability distribution.
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    # ===================================
    # NODE CONSTRUCTION
    # ===================================

    def __init__(self, arg1=None, arg2=None):
        if isinstance(arg1, str) and isinstance(arg2, ProbDistribution):
            node_id = arg1
            distrib = arg2
            """
            Creates a new chance node, with the given identifier and probability distribution

            :param node_id: the unique node identifier
            :param distrib: the probability distribution for the node
            """
            super(ChanceNode, self).__init__(node_id)
            if distrib.get_variable() != node_id:
                self.log.warning(node_id + "  != " + distrib.get_variable())
            self._distrib = distrib
            self._cached_values = None
        elif isinstance(arg1, str) and isinstance(arg2, Value):
            node_id = arg1
            value = arg2
            """
            Creates a change node with a unique value (associated with a probability 1.0)

            :param node_id: the unique node identifier
            :param value: the single value for the node
            """
            super(ChanceNode, self).__init__(node_id)
            self._distrib = SingleValueDistribution(node_id, value)
            self._cached_values = None
        else:
            raise NotImplementedError("UNDEFINED PARAMETERS")

    @dispatch(ProbDistribution)
    def set_distrib(self, distrib):
        """
        Sets the probability distribution of the node, and erases the existing one.

        :param distrib: the distribution for the node
        """
        self._distrib = distrib
        if distrib.get_variable() != self._node_id:
            self.log.warning(self._node_id + "  != " + distrib.get_variable())
            raise ValueError()

        self._cached_values = None

    @dispatch(BNode)
    def add_input_node(self, input_node):
        """
        Adds a new (input) relation for the node.

        :param input_node: the input node to connect
        """
        super(ChanceNode, self).add_input_node(input_node)

    @dispatch(str)
    def set_id(self, new_id):
        """
        Replaces the node identifier with a new one.

        :param new_id: the new identifier
        """
        old_id = self._node_id
        super(ChanceNode, self).set_id(new_id)
        self._distrib.modify_variable_id(old_id, new_id)

    @dispatch(float)
    def prune_values(self, threshold):
        """
        Prune the values with a probability below a given threshold.

        :param threshold: the probability threshold
        """
        if self._distrib.prune_values(threshold):
            self._cached_values = None

    # ===================================
    # GETTERS
    # ===================================

    @dispatch(Value)
    def get_prob(self, node_value):
        """
        Returns the probability associated with a specific value, according to the
        current distribution.

        The method assumes that the node is conditionally independent of every other
        node. If it isn't, one should use the getProb(condition, nodeValue) method
        instead.

        NB: the method should *not* be used to perform sophisticated inference, as it
        is not optimised and might lead to distorted results for very dependent
        networks

        :param node_value: the value for the node
        :return: its probability
        """
        if isinstance(self._distrib, IndependentDistribution):
            return self._distrib.get_prob(node_value)

        combinations = self.get_possible_conditions()
        total_prob = 0.
        for condition in combinations:
            prob = 1.
            for input_node in self._input_nodes:
                if isinstance(input_node, ChanceNode):
                    value = condition.get_value(input_node.get_id())
                    prob *= input_node.get_prob(value)

            total_prob += prob * self._distrib.get_prob(condition, node_value)

        return total_prob

    @dispatch(Assignment, Value)
    def get_prob(self, condition, node_value):
        """
        Returns the probability associated with the conditional assignment and the
        node value, if one is defined.

        :param condition: the condition
        :param node_value: the value for the node
        :return: the associated probability
        """
        try:
            return self._distrib.get_prob(condition, node_value)
        except Exception as e:
            self.log.warning("exception: %s" % e)
            raise ValueError()

    @dispatch()
    def sample(self):
        """
        Returns a sample value for the node, according to the probability distribution
        currently defined.

        The method assumes that the node is conditionally independent of every other
        node. If it isn't, one should use the sample(condition) method instead.

        :return: the sample value
        """
        if isinstance(self._distrib, IndependentDistribution):
            return self._distrib.sample()

        input_sample = Assignment()
        for input_node in self._input_nodes.values():
            if isinstance(input_node, ChanceNode) or isinstance(input_node, ActionNode):
                input_sample.add_pair(input_node.get_id(), input_node.sample())

        return self.sample(input_sample)

    @dispatch(Assignment)
    def sample(self, condition):
        """
        Returns a sample value for the node, given a condition. The sample is selected
        according to the probability distribution for the node.

        :param condition: the value assignment on conditional nodes
        :return: the sample value
        """
        if isinstance(self._distrib, IndependentDistribution):
            return self._distrib.sample()
        else:
            return self._distrib.sample(condition)

    @dispatch()
    def get_values(self):
        """
        Returns a discrete set of values for the node. If the variable for the node
        has a continuous range, this set if based on a discretisation procedure
        defined by the distribution.

        :return: the discrete set of values
        """
        if self._cached_values is None:
            self._cached_values = self._distrib.get_values()

        return self._cached_values

    @dispatch()
    def get_nb_values(self):
        """
        Returns the number of values for the node.

        :return: the number of values
        """
        if isinstance(self._distrib, ContinuousDistribution):
            return Settings.discretization_buckets
        else:
            return len(self.get_values())

    @dispatch()
    def get_distrib(self):
        """
        Returns the probability distribution attached to the node.

        :return: the distribution
        """
        return self._distrib

    @dispatch()
    def get_factor(self):
        """
        Returns the "factor matrix" mapping assignments of conditional variables + the
        node variable to a probability value.

        :return: the factor matrix
        """
        factor = dict()
        conditions = self.get_possible_conditions()

        for condition in conditions:
            posterior = self._distrib.get_prob_distrib(condition)
            for value in posterior.get_values():
                factor[Assignment(condition, self._node_id, value)] = posterior.get_prob(value)

        return factor

    # ===================================
    # UTILITIES
    # ===================================

    def __copy__(self):
        """
        Returns a copy of the node. Note that only the node content is copied, not its
        connection with other nodes.

        :return: the copy
        """
        chance_node = ChanceNode(self._node_id, copy(self._distrib))
        if self._cached_values is not None:
            chance_node._cached_values = copy(self._cached_values)
        return chance_node

    def __hash__(self):
        """
        Returns the hashcode for the node (based on the hashcode of the identifier and
        the distribution).

        :return: the hashcode for the node
        """
        return super(ChanceNode, self).__hash__() + self._distrib.__hash__()

    def __str__(self):
        """
        Returns the string representation of the distribution.

        :return: the string representation for the distribution
        """
        return self._distrib.__str__()

    __repr__ = __str__

    # ===================================
    # PRIVATE AND PROTECTED METHODS
    # ===================================

    @dispatch(str, str)
    def modify_variable_id(self, old_id, new_id):
        """
        Modify the variable identifier with new identifier.

        :param old_id: the old identifier for the node
        :param new_id: the new identifier for the node
        """
        super(ChanceNode, self).modify_variable_id(old_id, new_id)
        self._distrib.modify_variable_id(old_id, new_id)
