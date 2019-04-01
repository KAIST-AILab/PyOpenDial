from multipledispatch import dispatch
import random
import logging

from bn.nodes.b_node import BNode
from bn.values.value import Value
from bn.values.value_factory import ValueFactory
from datastructs.assignment import Assignment


class ActionNode(BNode):
    """
    Representation of an action node (sometimes also called decision node). An action
    node is defined as a set of mutually exclusive action values.
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
            Creates a new action node with a unique identifier, and no values

            :param node_id: the node identifier
            """
            super(ActionNode, self).__init__(node_id)
            self._action_values = set()
            self._action_values.add(ValueFactory.none())
        elif isinstance(arg1, str) and isinstance(arg2, set):
            node_id = arg1
            action_values = arg2
            """
            Creates a new action node with a unique identifier and a set of values
            :param node_id: the node identifier
            :param action_values: the values for the action
            """
            super(ActionNode, self).__init__(node_id)
            self._action_values = set()
            self._action_values.update(action_values)
        else:
            raise NotImplementedError("UNDEFINED PARAMETERS")

    @dispatch(BNode)
    def add_input_node(self, input_node):
        self.log.warning("Action node cannot have any input nodes, ignoring call")
        raise ValueError()

    @dispatch(Value)
    def add_value(self, value):
        """
        Adds a new action values to the node.

        :param input_node: the value to add
        """

        self._action_values.add(value)

    @dispatch(set)
    def add_values(self, values):
        """
        Adds a set of action values to the node.

        :param value: the values to add
        """

        for value in values:
            self.add_value(value)

    @dispatch(Value)
    def remove_value(self, value):
        """
        Removes a value from the action values set.

        :param value: the value to remove
        """
        self._action_values.remove(value)

    @dispatch(set)
    def remove_values(self, values):
        """
        Removes a set of values from the action values.

        :param values: the values to remove
        """
        self._action_values.difference_update(values)

    @dispatch()
    def get_factor(self):
        """
        Returns the factor matrix for the action node. The matrix lists the possible
        actions for the node, along with a uniform probability distribution over its values.

        :return: the factor matrix corresponding to the node
        """
        factor = dict()
        for action_value in self._action_values:
            factor[Assignment(self._node_id, action_value)] = 1.0 / len(self._action_values)
        return factor

    @dispatch(Value)
    def get_prob(self, action_value):
        """
        Returns a probability uniformly distributed on the alternative values.

        :param action_value: the value to check
        :return: 1/|values|
        """
        # TODO: check bug > Not using method argument.
        return 1. / len(self._action_values)

    # ===================================
    # GETTERS
    # ===================================

    @dispatch()
    def get_values(self):
        """
        Returns the list of values currently listed in the node.

        :return: the list of values
        """
        return set(self._action_values)

    @dispatch()
    def sample(self):
        """
        Returns a sample point for the action, assuming a uniform distribution over the action values.

        :return: the sample value
        """
        idx = random.randint(0, len(self._action_values) - 1)
        return list(self._action_values)[idx]

    # ===================================
    # UTILITIES
    # ===================================

    def __copy__(self):
        """
        Copies the action node. Note that only the node content is copied, not its connection with other nodes.

        :return: the copy of the node
        """
        return ActionNode(self._node_id, self._action_values)

    def __str__(self):
        """
        Returns a string representation of the node, which states the node identifier followed by the action values.

        :return: the string of the node information
        """
        return self._node_id + ": [" + ', '.join([str(action_value) for action_value in self._action_values]) + ']'

    __repr__ = __str__

    def __hash__(self):
        """
        Returns the hashcode corresponding to the action node.

        :return: the hashcode
        """
        return hash(self._node_id) + hash(frozenset(self._action_values))

    @dispatch(set)
    def set_values(self, new_values):
        """
        Set the action values to new values.

        :param new_values: the list of new values
        """
        self._action_values = new_values
