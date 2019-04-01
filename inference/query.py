import logging
from collections import Collection

from multipledispatch import dispatch

from bn.b_network import BNetwork
from bn.nodes.utility_node import UtilityNode
from datastructs.assignment import Assignment


class Query:
    """
    Representation of an inference query, which can be either a probability query, a
    utility query, or a reduction query.
    """

    log = logging.getLogger('PyOpenDial')

    def __init__(self, network, query_vars, evidence):
        if isinstance(network, BNetwork) and isinstance(query_vars, Collection) and isinstance(evidence, Assignment):
            self._network = network
            self._query_vars = set(query_vars)
            self._evidence = evidence

            if len(query_vars) == 0:
                self.log.warning("empty set of query variables: " + self.__str__())
            elif not network.get_node_ids().issuperset(query_vars):
                self.log.warning("mismatch between query variables and network nodes: " + str(query_vars) + " not included in " + str(network.get_node_ids()))
        else:
            raise NotImplementedError("UNDEFINED PARAMETERS")

    def __str__(self):
        results = []
        if isinstance(self, ProbQuery):
            results.append('P(')
        elif isinstance(self, UtilQuery):
            results.append('U(')
        else:
            results.append('Reduce(')

        for idx, query_var in enumerate(self._query_vars):
            results.append(query_var)
            if idx < len(self._query_vars) - 1:
                results.append(',')

        if not self._evidence.is_empty():
            results.append('|')
            results.append(str(self._evidence))

        results.append(')')

        return ''.join(results)

    def __hash__(self):
        return hash(self._query_vars) + 2 * hash(self._evidence)

    def get_network(self):
        """
        Returns the Bayesian network for the query.

        :return the Bayesian network.
        """
        return self._network

    def get_query_vars(self):
        """
        Returns the query variables.

        :return: the query variables.
        """
        return self._query_vars

    def get_evidence(self):
        """
        Returns the evidence for the query.

        :return: the evidence.
        """
        return self._evidence

    def get_filtered_sorted_nodes(self):
        """
        Returns a list of nodes sorted according to the ordering in
        BNetwork.getSortedNodes() and pruned from the irrelevant nodes

        :return: the ordered list of relevant nodes
        """
        filtered_nodes = list()
        irrelevant_nodes = self._get_irrelevant_nodes()
        for node in self._network.get_sorted_nodes():
            if node.get_id() not in irrelevant_nodes:
                filtered_nodes.append(node)

        return filtered_nodes

    def _get_irrelevant_nodes(self):
        """
        Assuming a particular query P(queryVars|evidence) or U(queryVars|evidence) on
        the provided Bayesian network, determines which nodes is relevant for the
        inference and which one can be discarded without affecting the final result.

        :return: irrelevant node ids.
        """
        irrelevant_node_ids = set()
        flag = True

        while flag:
            flag = False
            for node_id in self._network.get_node_ids():
                node = self._network.get_node(node_id)

                if node_id not in irrelevant_node_ids and irrelevant_node_ids.issuperset(node.get_output_node_ids()) and node_id not in self._query_vars and not self._evidence.contains_var(node_id) and not isinstance(node, UtilityNode):
                    irrelevant_node_ids.add(node_id)
                    flag = True
                    break
                elif not isinstance(self, UtilQuery) and node_id not in irrelevant_node_ids and isinstance(node, UtilityNode):
                    irrelevant_node_ids.add(node_id)
                    flag = True
                    break

        return irrelevant_node_ids

    def get_sorted_query_vars(self):
        """
        Returns the query variables in sorted order (from the base to the leaves)

        :return: the ordered query variables
        """
        results = list()
        for node in self._network.get_sorted_nodes():
            if node.get_id() in self._query_vars:
                results.append(node.get_id())

        results = list(reversed(results))
        return results


class ProbQuery(Query):
    """
    Representation of a probability query P(queryVars | evidence) on a specific
    Bayesian network.
    """
    def __init__(self, network, query_vars, evidence):
        if not isinstance(network, BNetwork) or not isinstance(query_vars, Collection) or not isinstance(evidence, Assignment):
            raise NotImplementedError("UNDEFINED PARAMETERS")

        super().__init__(network, query_vars, evidence)


class UtilQuery(Query):
    """
    Representation of an utility query U(queryVars | evidence) on a specific
    Bayesian network.
    """
    def __init__(self, network, query_vars, evidence):
        if not isinstance(network, BNetwork) or not isinstance(query_vars, Collection) or not isinstance(evidence, Assignment):
            raise NotImplementedError("UNDEFINED PARAMETERS")
        super().__init__(network, query_vars, evidence)


class ReduceQuery(Query):
    """
    Representation of a reduction Query where the Bayesian network is reduced to a
    new network containing only the variables queryVars, and integrating the
    evidence.
    """
    def __init__(self, network, query_vars, evidence):
        if not isinstance(network, BNetwork) or not isinstance(query_vars, Collection) or not isinstance(evidence, Assignment):
            raise NotImplementedError("UNDEFINED PARAMETERS")
        super().__init__(network, query_vars, evidence)
