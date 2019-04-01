from multipledispatch import dispatch
import abc
import functools
import logging
from collections import Collection

from regex.regex import Pattern

from datastructs.value_range import ValueRange
from utils.string_utils import StringUtils


class BNodeWrapper:
    pass


@functools.total_ordering
class BNode(BNodeWrapper):
    """
    Basic representation of a node integrated in a Bayesian Network. The node is
    defined via a unique identifier and a set of incoming and outgoing relations with
    other nodes.

    The class is abstract -- each node needs to be instantiated in a concrete subclass
    such as ChanceNode, ActionNode or UtilityNode.
    """
    __metaclass__ = abc.ABCMeta

    # logger
    log = logging.getLogger('PyOpenDial')

    # ===================================
    # NODE CONSTRUCTION
    # ===================================

    def __init__(self, node_id):
        if not isinstance(node_id, str):
            raise NotImplementedError("UNDEFINED PARAMETERS")
        """
        Creates a node with a unique identifier, and a empty set of incoming nodes.

        :param node_id: the identifier for the node
        """
        self._node_id = node_id
        self._input_nodes = dict()
        self._output_nodes = dict()
        self._network = None

    @dispatch(BNodeWrapper)
    def add_input_node(self, input_node):
        """
        Adds a new relation from the node given as argument to the current node.

        :param input_node: the node to add
        """
        if input_node is self:
            self.log.warning("cannot add itself: " + self._node_id)
            raise ValueError()

        if self._contains_cycles(input_node):
            self.log.warning("there is a cycle between " + input_node.get_id() + " and " + self._node_id)
            raise ValueError()

        from bn.nodes.action_node import ActionNode
        if isinstance(self, ActionNode):
            self.log.warning("an action node cannot be dependent on any other node")
            raise ValueError()

        from bn.nodes.utility_node import UtilityNode
        if isinstance(input_node, UtilityNode):
            self.log.warning("an utility node cannot be the input of any other node ("
                             + input_node.get_id() + " -> " + self._node_id + ")")
            raise ValueError()

        self._add_input_node_internal(input_node)
        input_node._add_output_node_internal(self)

    @dispatch(Collection)
    def add_input_nodes(self, input_nodes):
        """
        Adds new relations from the nodes given as arguments to the current node.

        :param input_nodes: the nodes to add
        """
        for input_node in input_nodes:
            self.add_input_node(input_node)

    @dispatch(str)
    def remove_input_node(self, input_node_id):
        """
        Removes a relation between an input node and the current node.

        :param input_node_id: the identifier for the incoming node to remove
        :return: true if a relation between the nodes existed, false otherwise
        """
        if input_node_id not in self._input_nodes:
            self.log.warning("node " + input_node_id + " is not an input node for " + self._node_id)
            raise ValueError()

        removal1 = input_node_id in self._input_nodes and self._input_nodes[input_node_id]._remove_output_node_internal(self._node_id)
        removal2 = self._remove_input_node_internal(input_node_id)

        if removal1 != removal2:
            self.log.warning("inconsistency between input and output links for "
                             + input_node_id + " and " + self._node_id)
            raise ValueError()

        return removal2

    @dispatch()
    def remove_input_nodes(self):
        """
        Removes all input nodes
        """
        for input_node in self._input_nodes.values():
            self.remove_input_node(input_node.get_id())

    @dispatch(str)
    def remove_output_node(self, output_node_id):
        """
        Removes a relation between an output node and the current node.

        :param output_node_id: the identifier for the outgoing node to remove
        :return: true if a relation between the nodes existed, false otherwise
        """
        if output_node_id in self._output_nodes:
            self.log.warning("node " + output_node_id + " is not an input node for " + self._node_id)
            raise ValueError()

        removal1 = output_node_id in self._output_nodes and self._output_nodes[output_node_id]._remove_input_node_internal(self._node_id)
        removal2 = self._remove_output_node_internal(output_node_id)

        if removal1 != removal2:
            self.log.warning("inconsistency between input and output links for "
                             + output_node_id + " and " + self._node_id)
            raise ValueError()

        return removal2

    @dispatch()
    def remove_all_relations(self):
        """
        Removes all input and output relations to the node.
        """
        for input_node in self._input_nodes.values():
            self.remove_input_node(input_node.get_id())

        for output_node in self._output_nodes.values():
            self.remove_output_node(output_node.get_id())

    @dispatch(str)
    def set_id(self, new_node_id):
        """
        Changes the identifier for the node.

        :param new_node_id: the new identifier
        """
        old_node_id = self._node_id
        self._node_id = new_node_id

        self.modify_variable_id(old_node_id, new_node_id)

        for input_node in self._input_nodes.values():
            input_node.modify_variable_id(old_node_id, new_node_id)

        for output_node in self._output_nodes.values():
            output_node.modify_variable_id(old_node_id, new_node_id)

        if self._network is not None:
            self._network.modify_variable_id(old_node_id, new_node_id)

    @dispatch(object)
    def set_network(self, network):
        """
        Sets the Bayesian network associated with the node (useful to inform the
        network of change of identifiers).

        :param network: the Bayesian network to associate to the node
        """
        self._network = network

    # ===================================
    # GETTERS
    # ===================================

    @dispatch()
    def get_id(self):
        """
        Returns the identifier for the node.

        :return: the node identifier
        """
        return self._node_id

    @dispatch(str)
    def has_input_node(self, node_id):
        """
        Returns true if the node contains an input node identified by the given id,
        and false otherwise.

        :param node_id: the identifier for the node
        :return: true if there is such input node, false otherwise
        """
        return node_id in self._input_nodes

    @dispatch(str)
    def has_output_node(self, node_id):
        """
        Returns true if the node contains an output node identified by the given id,
        and false otherwise.

        :param param: the identifier for the node
        :return: true if there is such output node, false otherwise
        """
        return node_id in self._output_nodes

    @dispatch()
    def get_input_nodes(self):
        """
        Returns the set of input nodes

        :return: the input nodes
        """
        return set(self._input_nodes.values())

    @dispatch(type)
    def get_input_nodes(self, cls):
        """
        Returns the set of input nodes of a certain class.

        :param cls: the class
        :return: the input nodes
        """
        result = set()
        for input_node in self._input_nodes.values():
            if isinstance(input_node, cls):
                result.add(input_node)

        return result

    @dispatch()
    def get_input_node_ids(self):
        """
        Returns the identifiers for the set of input nodes.

        :return: the ids for the input nodes
        """
        return set(self._input_nodes.keys())

    @dispatch()
    def get_output_nodes(self):
        """
        Returns the set of output nodes

        :return: the output nodes
        """
        return set(self._output_nodes.values())

    @dispatch(type)
    def get_output_nodes(self, cls):
        """
        Returns the set of output nodes of a certain class.

        :param cls: the class
        :return: the input nodes
        """
        result = set()
        for output_node in self._output_nodes.values():
            if isinstance(output_node, cls):
                result.add(output_node)

        return result

    @dispatch()
    def get_output_node_ids(self):
        """
        Returns the identifiers for the set of output nodes.

        :return: the ids for the input nodes
        """
        return set(self._output_nodes.keys())

    @dispatch()
    def get_ancestors(self):
        """
        Returns an ordered list of nodes which are the ancestors (via the relations)
        of the current node. The ordering puts the closest ancestors at the beginning
        of the list, and the most remote ancestors at the end.

        :return: an ordered list of ancestors for the node
        """
        ancestors = list()
        nodes_to_process = list()
        nodes_to_process.append(self)

        while len(nodes_to_process) > 0:
            cur_node = nodes_to_process.pop(0)
            for ancestor_node in cur_node.get_input_nodes():
                if ancestor_node not in ancestors:
                    ancestors.append(ancestor_node)

                if ancestor_node not in nodes_to_process:
                    nodes_to_process.append(ancestor_node)

        return ancestors

    @dispatch()
    def get_ancestor_ids(self):
        """
        Returns an order list of node identifiers which are the ancestors (via the
        relations) of the current node.

        :return: the ordered list of ancestor identifiers
        """
        ancestors = self.get_ancestors()
        return set([node.get_id() for node in ancestors])

    @dispatch(Collection)
    def get_ancestor_ids(self, variables_to_retain):
        """
        Returns the list of closest ancestors for the node among a set of possible
        variables. The variables not mentioned are ignored.

        :param variables_to_retain: the set of all variables from which to seek possible ancestors
        :return: the set of dependencies for the given variable
        """
        ancestors = set()
        nodes_to_process = [node for node in self.get_input_nodes()]

        while len(nodes_to_process) > 0:
            input_node = nodes_to_process.pop()
            # TODO: check bug > Is this condition right?
            if input_node.get_id() in variables_to_retain:
                ancestors.add(input_node.get_id())
            else:
                nodes_to_process.extend(input_node.get_input_nodes())

        return ancestors

    @dispatch(Collection)
    def get_descendant_ids(self, variables_to_retain):
        """
        Returns the list of closest descendants for the node among a set of possible
        variables. The variables not mentioned are ignored.

        :param variables_to_retain: the set of all variables from which to seek possible descendants
        :return: the set of relevant descendatns for the given variable
        """
        descendants = set()
        nodes_to_process = [node for node in self.get_output_nodes()]

        while len(nodes_to_process) > 0:
            output_node = nodes_to_process.pop()
            if output_node.get_id() in variables_to_retain:
                descendants.add(output_node.get_id())
            else:
                nodes_to_process.extend(output_node.get_output_nodes())

        return descendants

    @dispatch()
    def get_descendants(self):
        """
        Returns an ordered list of nodes which are the descendants (via the relations)
        of the current node. The ordering puts the closest descendants at the
        beginning of the list, and the most remote descendants at the end.

        :return: an ordered list of descendants for the node
        """
        descendants = list()
        nodes_to_process = list()
        nodes_to_process.append(self)

        while len(nodes_to_process) > 0:
            cur_node = nodes_to_process.pop(0)
            for descendant_node in cur_node.get_output_nodes():
                if descendant_node not in descendants:
                    descendants.append(descendant_node)

                if descendant_node not in nodes_to_process:
                    nodes_to_process.append(descendant_node)

        return descendants

    @dispatch()
    def get_descendant_ids(self):
        """
        Returns an order list of node identifiers which are the descendants (via the
        relations) of the current node.

        :return: the ordered list of descendant identifiers
        """
        descendants = self.get_descendants()
        return [node.get_id() for node in descendants]

    @dispatch(set)
    def has_descendant(self, variables):
        """
        Returns true if at least one of the variables given as argument is a
        descendant of this node, and false otherwise

        :param variables: the node identifiers of potential descendants
        :return: true if a descendant is found, false otherwise
        """
        nodes_to_process = list()
        nodes_to_process.append(self)

        # NB: we try to avoid recursion for efficiency reasons, and use a while loop instead
        while len(nodes_to_process) > 0:
            cur_node = nodes_to_process.pop(0)
            for descendant_node in cur_node.get_output_nodes():
                if descendant_node.get_id() in variables:
                    return True
                if descendant_node not in nodes_to_process:
                    nodes_to_process.append(descendant_node)

        return False

    @dispatch(set)
    def has_ancestor(self, variables):
        """
        Returns true if at least one of the variables given as argument is an ancestor
        of this node, and false otherwise.

        :param variables: the node identifiers of potential descendants
        :return: true if a descendant is found, false otherwise
        """
        nodes_to_process = list()
        nodes_to_process.append(self)

        while len(nodes_to_process) > 0:
            cur_node = nodes_to_process.pop(0)
            for ancestor_node in cur_node.get_input_nodes():
                if ancestor_node.get_id() in variables:
                    return True
                if ancestor_node not in nodes_to_process:
                    nodes_to_process.append(ancestor_node)

        return False

    @dispatch(Pattern)
    def has_descendant(self, pattern):
        """
        Returns true if at there exists at least one descendant whose identifier
        matches the regular expression pattern, and false otherwise

        :param pattern: the regular expression pattern to look for
        :return: true if a descendant is found, false otherwise
        """
        nodes_to_process = list()
        nodes_to_process.append(self)

        # NB: we try to avoid recursion for efficiency reasons, and use a while loop instead
        while len(nodes_to_process) > 0:
            cur_node = nodes_to_process.pop(0)
            for descendant_node in cur_node.get_output_nodes():
                if pattern.match(descendant_node.get_id()) is not None:
                    return True
                if descendant_node not in nodes_to_process:
                    nodes_to_process.append(descendant_node)

        return False

    @dispatch(set)
    def has_output_node(self, variables):
        """
        Returns true if the node has at least one output node in the set of
        identifiers provided as argument

        :param variables: the variable identifiers to check
        :return: true if at least one variable is an output node, false otherwise
        """
        for output_node in self.get_output_node_ids():
            if output_node in variables:
                return True

        return False

    @dispatch()
    @abc.abstractmethod
    def get_values(self):
        """
        Returns the set of distinct values that the node can take. The nature of those
        values depends on the node type.

        :return: the set of distinct values
        """
        raise NotImplementedError

    @dispatch()
    @abc.abstractmethod
    def get_factor(self):
        """
        Return the factor matrix associated with the node. The factor matrix is
        derived from the probability or utility distribution.

        :return: the factor matrix for the node
        """
        raise NotImplementedError

    @dispatch()
    def get_clique(self):
        """
        Returns the (maximal) clique in the network that contains this node.

        :return: the maximal clique
        """
        clique = set()
        clique.add(self._node_id)

        nodes_to_process = list()
        nodes_to_process.extend(self._input_nodes.values())
        nodes_to_process.extend(self._output_nodes.values())

        while len(nodes_to_process) > 0:
            cur_node = nodes_to_process.pop()
            clique.add(cur_node.get_id())

            for input_node in cur_node.get_input_nodes():
                if input_node.get_id() not in clique:
                    nodes_to_process.append(input_node)

            for output_node in cur_node.get_output_nodes():
                if output_node.get_id() not in clique:
                    nodes_to_process.append(output_node)

        return clique

    @dispatch()
    def get_possible_conditions(self):
        """
        Returns the list of possible assignment of input values for the node. If the
        node has no input, returns a list with a single, empty assignment.

        :return: the (unordered) list of possible conditions.
        """
        possible_input_values = ValueRange()

        for input_node in self._input_nodes.values():
            possible_input_values.add_values(input_node.get_id(), input_node.get_values())

        return possible_input_values.linearize()

    # ===================================
    # UTILITIES
    # ===================================

    @abc.abstractmethod
    def __copy__(self):
        """
        Creates a copy of the current node. Needs to be instantiated by the concrete subclasses.

        :return: the copy of the node
        """
        raise NotImplementedError()

    def __hash__(self):
        """
        Returns the hashcode, simply defined as the hashcode of the identifier.

        :return: the hashcode for the identifier
        """
        return hash(self._node_id)

    def __eq__(self, other):
        """
        Returns true if the given argument is a node with identical identifier.

        :param other: the object to compare
        :return: true if the argument is a node with an identical identifier
        """
        if not isinstance(other, BNode):
            return False

        return self._node_id == other.get_id() and self.get_input_nodes() == other.get_input_nodes()

    def __lt__(self, other):
        """
        Compares the node to other nodes, in order to derive the topological order of
        the network. If the node given as argument is one ancestor of this node,
        return -100. If the opposite is true, returns +100. Else, returns the
        difference between the size of the respective ancestors lists. Finally, if
        both lists are empty, returns +1 or -1 depending on the lexicographic order of
        the node identifiers.

        :param other: the node to compare
        :return: the comparison result
        """
        if len(self.get_input_node_ids()) == 0 and len(other.get_input_node_ids()) > 0:
            return False

        if len(self.get_input_node_ids()) > 0 and len(other.get_input_node_ids()) == 0:
            return True

        if len(self.get_input_node_ids()) == 0 and len(other.get_input_node_ids()) == 0:
            from bn.nodes.action_node import ActionNode
            if isinstance(self, ActionNode) and not isinstance(other, ActionNode):
                return False

            if not isinstance(self, ActionNode) and isinstance(other, ActionNode):
                return True

            return StringUtils.compare(self._node_id, other.get_id()) < 0

        ancestors = self.get_ancestors()
        if other in ancestors:
            return True

        other_ancestors = other.get_ancestors()
        if self in other_ancestors:
            return False

        size_diff = len(other_ancestors) - len(ancestors)
        if size_diff != 0:
            return True if size_diff < 0 else False

        return StringUtils.compare(self._node_id, other.get_id()) < 0

    def __str__(self):
        """
        Returns the string identifier for the node.

        :return: the string identifier for the node
        """
        return self._node_id

    # ===================================
    # PROTECTED AND PRIVATE METHODS
    # ===================================

    @dispatch(str, str)
    def modify_variable_id(self, old_node_id, new_node_id):
        """
        Replaces the identifier for the input and output nodes with the new identifier.

        :param old_node_id: the old label for the node
        :param new_node_id: the new label for the node
        """
        if old_node_id in self._input_nodes:
            input_node = self._input_nodes[old_node_id]
            self._remove_input_node_internal(old_node_id)
            self._add_input_node_internal(input_node)
        elif old_node_id in self._output_nodes:
            output_node = self._output_nodes[old_node_id]
            self._remove_output_node_internal(old_node_id)
            self._add_output_node_internal(output_node)

    @dispatch(BNodeWrapper)
    def _add_input_node_internal(self, input_node):
        """
        Adds a new incoming relation to the node. This method should never be called
        outside the addRelation method, to ensure consistency between the input and
        output links.

        :param input_node: the input node to add
        """
        if input_node.get_id() in self._input_nodes:
            # TODO: check refactor > Does it need to raise an exception?
            self.log.warning("node " + input_node.getId()
                             + " already included in the input nodes of " + self._node_id)
            raise ValueError()

        self._input_nodes[input_node.get_id()] = input_node

    @dispatch(BNodeWrapper)
    def _add_output_node_internal(self, output_node):
        """
        Adds a new outgoing relation to the node. This method should never be called
        outside the addRelation method, to ensure consistency between the input and
        output links.

        :param output_node: the output node to add
        """
        if output_node.get_id() in self._output_nodes:
            # TODO: check refactor > Does it need to raise an exception?
            self.log.warning("node " + output_node.get_id()
                             + " already included in the output nodes of " + self._node_id)
            raise ValueError()
        else:
            self._output_nodes[output_node.get_id()] = output_node

    @dispatch(str)
    def _remove_input_node_internal(self, input_node_id):
        """
        Removes an outgoing relation to the node. This method should never be called
        outside the removeRelation method, to ensure consistency between the input and
        output links.

        :param input_node_id: the input node to remove
        :return: true if a relation between the nodes existed, false otherwise
        """
        if input_node_id not in self._input_nodes:
            return False

        del self._input_nodes[input_node_id]
        return True

    @dispatch(str)
    def _remove_output_node_internal(self, output_node_id):
        """
        Removes an outgoing relation to the node. This method should never be called
        outside the removeRelation method, to ensure consistency between the input and
        output links.

        :param output_node_id: the output node to remove
        :return: true if a relation between the nodes existed, false otherwise
        """
        if output_node_id not in self._output_nodes:
            # TODO: check refactor > Does it need to raise an exception?
            self.log.warning("node " + output_node_id + " is not an output node for " + self._node_id)
            raise ValueError()

        output_node = self._output_nodes.pop(output_node_id)
        return output_node is not None

    @dispatch(BNodeWrapper)
    def _contains_cycles(self, input_node):
        """
        Checks whether a cycle exists between the given node and the present one.

        :param input_node: the input node
        :return: true if such a cycle exists, false otherwise
        """
        if input_node in self.get_descendants():
            return True

        return False
