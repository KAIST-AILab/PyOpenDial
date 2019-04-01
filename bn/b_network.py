import logging
from collections import Collection
from copy import copy

from multipledispatch import dispatch

from bn.nodes.action_node import ActionNode
from bn.nodes.b_node import BNode
from bn.nodes.chance_node import ChanceNode
from bn.nodes.utility_node import UtilityNode


class BNetworkWrapper:
    pass


class BNetwork(BNetworkWrapper):
    """
    Representation of a Bayesian Network augmented with value and action nodes. The
    network is simply defined as a set of nodes connected with each other.
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    # ===================================
    # NETWORK CONSTRUCTION
    # ===================================

    def __init__(self, arg1=None):
        if arg1 is None:
            """
            Constructs an empty network
            """
            self._nodes = dict()
            self._chance_nodes = dict()
            self._utility_nodes = dict()
            self._action_nodes = dict()
        elif isinstance(arg1, Collection):
            nodes = arg1
            """
            Creates a network with the provided list of nodes

            :param nodes: the nodes to add
            """
            self.__init__()
            self.add_nodes(nodes)
        else:
            raise NotImplementedError()

    @dispatch(BNode)
    def add_node(self, node):
        """
        Adds a new node to the network. Note: if the node already exists, it is better
        to use the "replaceNode" method, to avoid warning messages.

        :param node: node to add
        """
        node_id = node.get_id()
        if node_id in self._nodes:
            self.log.warning("network already contains a node with identifier " + node.get_id())
            raise ValueError()

        self._nodes[node_id] = node
        node.set_network(self)

        if isinstance(node, ChanceNode):
            self._chance_nodes[node_id] = node
        elif isinstance(node, UtilityNode):
            self._utility_nodes[node_id] = node
        elif isinstance(node, ActionNode):
            self._action_nodes[node_id] = node

    @dispatch(Collection)  # collection of BNode
    def add_nodes(self, nodes):
        """
        Add a collection of new nodes to the network.

        :param nodes: the collection of nodes to add
        """
        for node in nodes:
            self.add_node(node)

    @dispatch(BNetworkWrapper)
    def add_network(self, network):
        """
        Adds all the nodes in the network provided as argument to the current network.

        :param network: the network to include
        """
        for node in network.get_nodes():
            if self.has_node(node.get_id()):
                self.remove_node(node.get_id())

        for node in network.get_nodes():
            self.add_node(copy(node))

        for old_node in network.get_nodes():
            new_node = self.get_node(old_node.get_id())
            for input_node_id in old_node.get_input_node_ids():
                new_input_node = self.get_node(input_node_id)
                new_node.add_input_node(new_input_node)

    @dispatch(BNode)
    def replace_node(self, node):
        """
        Replaces an existing node with a new one (with same identifier).

        :param node: the new value for the node
        """
        if node.get_id() not in self._nodes:
            self.log.debug("network does not contain a node with identifier " + node.get_id())
            raise ValueError()

        self.remove_node(node.get_id())
        self.add_node(node)

    @dispatch(str)
    def remove_node(self, node_id):
        """
        Removes a node from the network, given its identifier.

        :param node_id: the node identifier
        :return: the value for the node, if it exists
        """
        if node_id not in self._nodes:
            raise ValueError()

        node = self._nodes[node_id]
        for input_node in node.get_input_nodes():
            node.remove_input_node(input_node.get_id())

        for output_node in node.get_output_nodes():
            output_node.remove_input_node(node_id)

        if isinstance(node, ChanceNode):
            del self._chance_nodes[node_id]
        elif isinstance(node, UtilityNode):
            del self._utility_nodes[node_id]
        elif isinstance(node, ActionNode):
            del self._action_nodes[node_id]

        return self._nodes.pop(node_id)

    @dispatch(Collection)  # collection of strings
    def remove_nodes(self, value_node_ids):
        """
        Remove all the specified nodes.

        :param value_node_ids: the dnoes to remove
        :return: the removed nodes
        """
        removed = list()
        for id in value_node_ids:
            node = self.remove_node(id)
            removed.append(node)

        return removed

    @dispatch(str, str)
    def modify_variable_id(self, old_node_id, new_node_id):  # TODO: why new_node_id not used?
        """
        Modifies the node identifier in the Bayesian Network.

        :param old_node_id: the old node identifier
        :param new_node_id: the new node identifier
        """
        node = self._nodes.pop(old_node_id, None)
        if old_node_id in self._chance_nodes:
            del self._chance_nodes[old_node_id]
        if old_node_id in self._utility_nodes:
            del self._utility_nodes[old_node_id]
        if old_node_id in self._action_nodes:
            del self._action_nodes[old_node_id]

        if node is None:
            self.log.warning("node " + old_node_id + " did not exist, cannot change its identifier")
            raise ValueError()

        self.add_node(node)

    @dispatch(BNetworkWrapper)
    def reset(self, network):
        """
        Resets the Bayesian network to only contain the nodes contained in the
        argument. Everything else is erased.

        :param network: the network that contains the nodes to include after the reset
        """
        if self is network:
            return

        self._nodes.clear()
        self._chance_nodes.clear()
        self._utility_nodes.clear()
        self._action_nodes.clear()

        for node in network.get_nodes():
            self.add_node(node)

    # ===================================
    # GETTERS
    # ===================================

    @dispatch(str)
    def has_node(self, node_id):
        """
        Returns true if the network contains a node with the given identifier.

        :param node_id: the node identifier
        :return: true if the node exists in the network, false otherwise
        """
        return node_id in self._nodes

    @dispatch(str)
    def get_node(self, node_id):
        """
        Returns the node associated with the given identifier in the network. If no
        such node is present, returns null.

        :param node_id: the node identifier
        :return: the node, if it exists, or null otherwise
        """
        if node_id not in self._nodes:
            self.log.critical("network does not contain a node with identifier " + node_id)
            raise ValueError()

        return self._nodes[node_id]

    @dispatch()
    def get_nodes(self):
        """
        Returns the collection of nodes currently in the network

        :return: the collection of nodes
        """
        return self._nodes.values()

    @dispatch(Collection)  # collection of strings
    def get_nodes(self, node_ids):
        """
        Returns the collection of nodes currently in the network by their ids

        :param node_ids: the ids
        :return: the resulting set of nodes
        """
        result = set()
        for node_id in node_ids:
            if node_id in self._nodes:
                result.add(self._nodes[node_id])

        return result

    @dispatch(type)
    def get_nodes(self, class_type):
        """
        Returns the set of nodes belonging to a certain class (extending BNode).

        :param class_type: the class
        :return: the resulting set of nodes
        """
        nodes_of_class = set()
        for node in self._nodes.values():
            if type(node) == class_type:
                nodes_of_class.add(node)

        return nodes_of_class

    @dispatch(str)
    def has_chance_node(self, node_id):
        """
        Returns true if the network contains a chance node with the given identifier,
        and false otherwise.

        :param node_id: the node identifier to check
        :return: true if a chance node is found, false otherwise
        """
        return node_id in self._chance_nodes

    @dispatch(Collection)  # collection of strings
    def has_chance_nodes(self, node_ids):
        """
        Returns true if the network contains chance nodes for all the given
        identifiers, and false otherwise.

        :param node_ids: the node identifiers to check
        :return: true if all the chance nodes is found, false otherwise
        """
        for node_id in node_ids:
            if node_id not in self._chance_nodes:
                return False

        return True

    @dispatch(str)
    def get_chance_node(self, node_id):
        """
        Returns the chance node associated with the identifier, if one exists. Else,
        returns null.

        :param node_id: the node identifier
        :return: the chance node
        """
        if node_id not in self._chance_nodes:
            self.log.warning("network does not contain a chance node with identifier " + node_id)
            raise ValueError()

        return self._chance_nodes[node_id]

    @dispatch()
    def get_chance_nodes(self):
        """
        Returns the collection of chance nodes currently in the network.

        :return: the collection of chance nodes
        """
        return set(self._chance_nodes.values())

    @dispatch()
    def get_chance_node_ids(self):
        """
        Returns the collection of chance node identifiers currently in the network.

        :return: the collection of identifiers of chance nodes
        """
        return set(self._chance_nodes.keys())

    @dispatch(type)
    def get_node_ids(self, class_type):
        """
        Returns the set of nodes belonging to a certain class (extending BNode).

        :param class_type: the class
        :return: the resulting set of node identifiers
        """
        result = set()
        for chance_node in self._chance_nodes.values():
            if type(chance_node.get_distrib()) == class_type:
                result.add(chance_node.get_id())

        for utility_node in self._utility_nodes.values():
            if type(utility_node.get_function()) == class_type:
                result.add(utility_node.get_id())

        return result

    @dispatch(set, type)
    def contains_distrib(self, node_ids, class_type):
        """
        Returns true if at least one of the nodes in nodeIds has a distribution of
        type cls. Else, returns false.

        :param node_ids: the node identifiers to check
        :param class_type: the distribution class
        :return: true if at least one node in nodeIds has a distribution of type cls, else false
        """
        return not node_ids.isdisjoint(self.get_node_ids(class_type))

    @dispatch(str)
    def has_action_node(self, node_id):
        """
        Returns true if the network contains an action node with the given identifier,
        and false otherwise.

        :param node_id: the node identifier to check
        :return: true if a action node is found, false otherwise
        """
        return node_id in self._action_nodes

    @dispatch(str)
    def get_action_node(self, node_id):
        """
        Returns the action node associated with the identifier, if one exists. Else,
        returns null.

        :param node_id: the node identifier
        :return: the action node
        """
        if node_id not in self._action_nodes:
            self.log.critical("network does not contain an action node with identifier " + node_id)
            raise ValueError()

        return self._action_nodes[node_id]

    @dispatch()
    def get_action_nodes(self):
        """
        Returns the collection of action nodes currently in the network.

        :return: the collection of action nodes
        """
        return self._action_nodes.values()

    @dispatch()
    def get_action_node_ids(self):
        """
        Returns the collection of action node identifiers currently in the network.

        :return: the collection of identifiers of action nodes
        """
        return set(self._action_nodes.keys())

    @dispatch(str)
    def has_utility_node(self, node_id):
        """
        Returns true if the network contains a utility node with the given identifier,
        and false otherwise.

        :param node_id: the node identifier to check
        :return: true if a utility node is found, false otherwise
        """
        return node_id in self._utility_nodes

    @dispatch(str)
    def get_utility_node(self, node_id):
        """
        Returns the utility node associated with the identifier, if one exists. Else,
        returns null.

        :param node_id: the node identifier
        :return: the utility node
        """
        if node_id not in self._utility_nodes:
            self.log.critical("network does not contain a utility node with identifier " + node_id)
            raise ValueError()

        return self._utility_nodes[node_id]

    @dispatch()
    def get_utility_nodes(self):
        """
        Returns the collection of utility nodes currently in the network

        :return: the collection of utility nodes
        """
        return self._utility_nodes.values()

    @dispatch()
    def get_utility_node_ids(self):
        """
        Returns the collection of utility node identifiers currently in the network.

        :return: the collection of identifiers of utility nodes
        """
        return set(self._utility_nodes.keys())

    @dispatch()
    def get_node_ids(self):
        """
        Returns the set of nodes belonging to a certain class (extending BNode).

        :return: the resulting set of node identifiers
        """
        return set(self._nodes.keys())

    @dispatch()
    def get_sorted_nodes(self):
        """
        Returns an ordered list of nodes, where the ordering is defined in the
        compareTo method implemented in BNode. The ordering will place end nodes (i.e.
        nodes with no outward edges) at the beginning of the list, and start nodes
        (nodes with no inward edges) at the end of the list.

        :return: the ordered list of nodes
        """
        result = list(self._nodes.values())
        result.sort()
        return result

    @dispatch()
    def get_sorted_node_ids(self):
        """
        Returns the ordered list of node identifiers (see method above).

        :return: the ordered list of node identifiers
        """
        result = list()
        for node in self.get_sorted_nodes():
            result.append(node.get_id())

        return result

    @dispatch()
    def get_cliques(self):
        """
        Returns the set of maximal cliques that compose this network. The cliques are
        collections of nodes such that each node in the clique is connect to all the
        other nodes in the clique but to no nodes outside the clique.

        :return: the collection of cliques for the network.
        """
        cliques = list()

        node_ids_to_process = list()
        node_ids_to_process.extend(list(self._nodes.keys()))
        while len(node_ids_to_process) > 0:
            node_id = node_ids_to_process.pop()
            clique = self._nodes.get(node_id).get_clique()
            cliques.append(clique)

            for node_id in clique:
                if node_id in node_ids_to_process:
                    node_ids_to_process.remove(node_id)

        cliques.sort(key=lambda clique: hash(frozenset(clique)))

        return cliques

    @dispatch(set)
    def get_cliques(self, node_ids):
        """
        Returns the set of maximal cliques that compose this network, if one only
        looks at the clique containing the given subset of node identifiers. The
        cliques are collections of nodes such that each node in the clique is connect
        to all the other nodes in the clique but to no nodes outside the clique.

        :param node_ids: the subset of node identifiers to use
        :return: the collection of cliques for the network.
        """
        cliques = list()

        node_ids_to_process = list()
        node_ids_to_process.extend(list(self._nodes.keys()))

        remaining_node_ids_to_process = list()
        for node_id in node_ids_to_process:
            if node_id in node_ids:
                remaining_node_ids_to_process.append(node_id)
        node_ids_to_process = remaining_node_ids_to_process

        while len(node_ids_to_process) > 0:
            node_id = node_ids_to_process.pop()
            clique = self._nodes.get(node_id).get_clique()
            cliques.append(clique)

            for node_id in clique:
                if node_id in node_ids_to_process:
                    node_ids_to_process.remove(node_id)

        cliques.sort(key=lambda clique: hash(frozenset(clique)))

        return cliques

    @dispatch(set)
    def is_clique(self, node_ids):
        """
        Returns true if the subset of node identifiers correspond to a maximal clique
        in the network, and false otherwise.

        :param node_ids: the subset of node identifiers
        :return: true if subsetIds corresponds to a maximal clique, false otherwise
        """
        if len(node_ids) == 0:
            return False

        node_id = node_ids.pop()
        node_ids.add(node_id)
        return self.has_node(node_id) and self.get_node(node_id).get_clique() == node_ids

    # ===================================
    # UTILITIES
    # ===================================

    def __hash__(self):
        """
        Returns the hashcode for the network, defined as the hashcode for the node
        identifiers in the network.

        :return: the hashcode for the network
        """
        return hash(frozenset(self._nodes.keys()))

    def __copy__(self):
        """
        Returns a copy of the Bayesian network.

        :return: the copy
        """
        result = BNetwork()
        nodes = self.get_sorted_nodes()
        nodes.sort(reverse=True)

        for node in nodes:
            copied_node = copy(node)
            for input_node in node.get_input_nodes():
                if not result.has_node(input_node.get_id()):
                    raise ValueError()

                copied_node.add_input_node(result.get_node(input_node.get_id()))

            result.add_node(copied_node)

        return result

    def __str__(self):
        """
        Returns a basic string representation for the network, defined as the set of
        node identifiers in the network.

        :return: the string representation
        """
        return str(list(self._nodes.keys()))

    def __eq__(self, other):
        """
        Returns true if the object is also a Bayesian network with exactly the same
        node identifiers.

        :param other: the object to compare
        :return: true if o is network with identical identifiers, false otherwise
        """
        if not isinstance(other, BNetwork):
            return False

        return self._nodes.keys() == other.get_node_ids()

    @dispatch()
    def pretty_print(self):
        """
        Returns a pretty print representation of the network, comprising both the node
        identifiers and the graph structure.

        :return: the pretty print representation
        """
        result = 'Nodes: %s\n' % self._nodes.keys()
        result += 'Edges: \n'
        for node in self._nodes.values():
            result += '\t%s --> %s' % (node.get_input_node_ids(), node.get_id())

        return result
