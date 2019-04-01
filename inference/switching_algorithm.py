import logging
from collections import Collection

from multipledispatch import dispatch

from bn.b_network import BNetwork
from bn.distribs.continuous_distribution import ContinuousDistribution
from bn.nodes.chance_node import ChanceNode
from datastructs.assignment import Assignment
from inference.approximate.sampling_algorithm import SamplingAlgorithm
from inference.exact.variable_elimination import VariableElimination
from inference.inference_algorithm import InferenceAlgorithm
from inference.query import ProbQuery, ReduceQuery, UtilQuery, Query


class SwitchingAlgorithm(InferenceAlgorithm):
    """
    Switching algorithms that alternates between an exact algorithm (variable
    elimination) and an approximate algorithm (likelihood weighting) depending on the
    query.
    
    The switching mechanism is defined via two thresholds:
    - one threshold on the maximum branching factor of the network
    - one threshold on the maximum number of combination of values in a node factor

    If one of these threshold is exceeded or if the Bayesian network contains a
    continuous distribution, the selected algorithm will be likelihood weighting.
    Variable elimination is selected in the remaining cases.
    """

    log = logging.getLogger('PyOpenDial')

    max_branching_factor = 10
    max_nr_values = 5000
    
    def __init__(self):
        self._ve = VariableElimination()
        self._lw = SamplingAlgorithm()

    @dispatch(BNetwork, Collection, Assignment)
    def query_prob(self, network, query_vars, evidence):
        return super().query_prob(network, query_vars, evidence)

    @dispatch(BNetwork, Collection)
    def query_prob(self, network, query_vars):
        return super().query_prob(network, query_vars)

    @dispatch(BNetwork, str, Assignment)
    def query_prob(self, network, query_var, evidence):
        return super().query_prob(network, query_var, evidence)

    @dispatch(BNetwork, str)
    def query_prob(self, network, query_var):
        return super().query_prob(network, query_var, Assignment())

    @dispatch(ProbQuery)
    def query_prob(self, query):
        """
        Selects the best algorithm for performing the inference on the provided
        probability query and return its result.

        :param query: the probability query
        :return: the inference result
        """
        algorithm = self.select_best_algorithm(query)
        return algorithm.query_prob(query)

    @dispatch(BNetwork, Collection, Assignment)
    def query_util(self, network, query_vars, evidence):
        return super().query_util(network, query_vars, evidence)

    @dispatch(BNetwork, Collection)
    def query_util(self, network, query_vars):
        return super().query_util(network, query_vars)

    @dispatch(BNetwork, str, Assignment)
    def query_util(self, network, query_var, evidence):
        return super().query_util(network, query_var, evidence)

    @dispatch(BNetwork, str)
    def query_util(self, network, query_var):
        return super().query_util(network, query_var)

    @dispatch(UtilQuery)
    def query_util(self, query):
        """
        Selects the best algorithm for performing the inference on the provided
        utility query and return its result.

        :param query: the utility query
        :return: the inference result
        """
        algorithm = self.select_best_algorithm(query)
        return algorithm.query_util(query)

    @dispatch(BNetwork, Collection, Assignment)
    def reduce(self, network, query_vars, evidence):
        return super().reduce(network, query_vars, evidence)

    @dispatch(BNetwork, Collection)
    def reduce(self, network, query_vars):
        return super().reduce(network, query_vars)

    @dispatch(ReduceQuery)
    def reduce(self, query):
        """
        Reduces a Bayesian network to a subset of variables. The method is divided in
        three steps:

        - The method first checks whether inference is necessary at all or whether
        the current network can be returned as it is.
        - If inference is necessary, the algorithm divides the network into cliques
        and performs inference on each clique separately.
        - Finally, if only one clique is present, the reduction selects the best
        algorithm and return the result of the reduction process.

        :param query: the reduction query
        :return: the reduced network
        """
        algorithm = self.select_best_algorithm(query)
        result = algorithm.reduce(query)
        return result

    @dispatch(Query)
    def select_best_algorithm(self, query):
        for node in query.get_filtered_sorted_nodes():
            if len(node.get_input_node_ids()) > SwitchingAlgorithm.max_branching_factor:
                return self._lw
            if isinstance(node, ChanceNode):
                if isinstance(node.get_distrib(), ContinuousDistribution):
                    return self._lw

                nr_values = node.get_nb_values()
                for chance_node in node.get_input_nodes(ChanceNode):
                    nr_values *= chance_node.get_nb_values()

                if nr_values > SwitchingAlgorithm.max_nr_values:
                    return self._lw

        return self._ve
