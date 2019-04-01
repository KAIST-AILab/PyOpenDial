from multipledispatch import dispatch
from copy import copy
import logging

from bn.distribs.utility_function import UtilityFunction
from bn.distribs.utility_table import UtilityTable
from bn.nodes.b_node import BNode
from datastructs.assignment import Assignment


class UtilityNode(BNode):
    """
    Representation of a utility node (sometimes also called utility node)
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    # ===================================
    # NODE CONSTRUCTION
    # ===================================

    def __init__(self, arg1=None, arg2=None):
        if isinstance(arg1, str) and arg2 is None:
            node_id = arg1
            """
            Creates a new utility node, with an empty utility distribution

            :param node_id: the node identifier
            """
            super(UtilityNode, self).__init__(node_id)
            self._distrib = UtilityTable()
        elif isinstance(arg1, str) and isinstance(arg2, UtilityFunction):
            node_id = arg1
            distrib = arg2
            """
            Creates a new utility node, with the given utility distribution.

            :param node_id: the node identifier
            :param distrib: the utility distribution
            """
            super(UtilityNode, self).__init__(node_id)
            self._distrib = distrib
        else:
            raise NotImplementedError("UNDEFINED PARAMETERS")

    @dispatch(Assignment, float)
    def add_utility(self, input, value):
        """
        Adds a new utility to the node, valid for the given assignment on the input nodes

        :param input: a value assignment on the input nodes
        :param value: the assigned utility
        """
        if isinstance(self._distrib, UtilityTable):
            self._distrib.set_util(input, value)
        else:
            self.log.warning("utility distribution is not a table, cannot add value")
            raise ValueError()

    @dispatch(Assignment)
    def remove_utility(self, input):
        """
        Removes the utility associated with the input assignment from the node.

        :param input: the input associated with the utility to be removed
        """
        if isinstance(self._distrib, UtilityTable):
            self._distrib.remove_util(input)
        else:
            self.log.warning("utility distribution is not a table, cannot remove value")
            raise ValueError()

    @dispatch(UtilityFunction)
    def set_distrib(self, distrib):
        """
        Sets the distribution of the node.

        :param distrib: the distribution for the node
        """
        self._distrib = distrib

    @dispatch(str)
    def set_id(self, new_node_id):
        """
        Sets the identifier of the node.

        :param new_node_id: the new identifier for the node
        """
        super(UtilityNode, self).set_id(new_node_id)
        self._distrib.modify_variable_id(self._node_id, new_node_id)

    # ===================================
    # GETTERS
    # ===================================

    @dispatch(Assignment)
    def get_utility(self, input):
        """
        Returns the utility associated with the specific assignment on the input
        variables of the node.

        :param input: the input assignment
        :return: the associated utility
        """
        return self._distrib.get_util(input)

    @dispatch()
    def get_values(self):
        """
        Returns an empty set (a utility node has no "value", only utilities).

        :return: the empty set
        """
        return set()

    @dispatch()
    def get_function(self):
        """
        Returns the utility distribution.

        :return: the utility distribution
        """
        return self._distrib

    @dispatch()
    def get_factor(self):
        """
        Returns the factor matrix associated with the utility node, which maps an
        assignment of input variable to a given utility.

        :return: the factor matrix
        """
        factor = dict()

        conditions = self.get_possible_conditions()
        for condition in conditions:
            factor[condition] = self._distrib.get_util(condition)

        return factor

    # ===================================
    # UTILITIES
    # ===================================

    def __copy__(self):
        """
        Returns a copy of the utility node. Note that only the node content is copied,
        not its connection with other nodes.

        :return: the copy
        """
        return UtilityNode(self._node_id, copy(self._distrib))

    def __str__(self):
        """
        Returns a string representation of the node, consisting of the node utility distribution.

        :return: the string representation of the node
        """
        return str(self._distrib)

    def __hash__(self):
        """
        Returns the hashcode for the value, computed from the node identifier and the distribution.

        :return: the hashcode
        """
        return hash(self._node_id) - hash(self._distrib)
