import logging
import traceback
from collections import Collection, Callable

from multipledispatch import dispatch

from bn.b_network import BNetwork
from bn.distribs.continuous_distribution import ContinuousDistribution
from bn.distribs.empirical_distribution import EmpiricalDistribution
from bn.distribs.utility_table import UtilityTable
from bn.nodes.chance_node import ChanceNode
from datastructs.assignment import Assignment
from inference.approximate.intervals import Intervals
from inference.approximate.likelihood_weighting import LikelihoodWeighting
from inference.inference_algorithm import InferenceAlgorithm
from inference.query import ProbQuery, UtilQuery, Query, ReduceQuery
from settings import Settings

dispatch_namespace = dict()


class SamplingAlgorithm(InferenceAlgorithm):
    """
    Sampling-based inference algorithm for Bayesian networks. The class provides a set
    of functionalities for performing inference operations based on a particular
    sampling algorithm (e.g. likelihood weighting).
    """
    log = logging.getLogger('PyOpenDial')

    def __init__(self, arg1=None, arg2=None):
        if arg1 is None and arg2 is None:
            """
            Creates a new likelihood weighting algorithm with the specified number of
            samples and sampling time

            :param nr_samples: the maximum number of samples to collect
            :param max_sampling_time: the maximum sampling time
            """
            self._nr_samples = Settings.nr_samples
            self._max_sampling_time = Settings.max_sampling_time
        elif isinstance(arg1, int) and isinstance(arg2, int):
            nr_samples = arg1
            max_sampling_time = arg2
            """
            Creates a new likelihood weighting algorithm with the specified number of
            samples and sampling time

            :param nr_samples: the maximum number of samples to collect
            :param max_sampling_time: the maximum sampling time
            """
            self._nr_samples = nr_samples
            self._max_sampling_time = max_sampling_time
        else:
            raise NotImplementedError("UNDEFINED PARAMETERS")

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
        Queries for the probability distribution of the set of random variables in the
        Bayesian network, given the provided evidence

        :param query: the full query
        :return: the resulting probability distribution failed
        """
        is_query = LikelihoodWeighting(query, self._nr_samples, self._max_sampling_time)
        samples = is_query.get_samples()
        return EmpiricalDistribution(samples)

    @staticmethod
    @dispatch(BNetwork, Collection, namespace=dispatch_namespace)
    def extract_sample(network, query_vars):
        """
        Extracts a unique (non reweighted) sample for the query.

        :param network: the network on which to extract the sample
        :param query_vars: the variables to extract
        :return: the extracted sample
        """
        query = ProbQuery(network, query_vars, Assignment())
        is_query = LikelihoodWeighting(query, 1, Settings.max_sampling_time)

        samples = is_query.get_samples()
        if len(samples) == 0:
            raise ValueError()
        else:
            return samples[0].get_trimmed(query.get_query_vars())

    @dispatch(UtilQuery)
    def query_util(self, query):
        """
        Queries for the utility of a particular set of (action) variables, given the
        provided evidence

        :param query: the full query
        :return: the utility distribution
        """
        try:
            is_query = LikelihoodWeighting(query, self._nr_samples, self._max_sampling_time)
            samples = is_query.get_samples()

            utility_table = UtilityTable()
            for sample in samples:
                utility_table.increment_util(sample, sample.get_utility())

            return utility_table
        except Exception as e:
            traceback.print_tb(e.__traceback__)
            return UtilityTable()

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

    @dispatch(BNetwork)
    def query_util(self, network):
        """
        Queries for the utility without any particular query variable

        :param network: the graphical model
        :return: the utility
        """
        query = UtilQuery(network, network.get_chance_node_ids(), Assignment())
        is_query = LikelihoodWeighting(query, self._nr_samples, self._max_sampling_time)

        samples = is_query.get_samples()

        # TODO: not implemented parallelized version.
        total = 0.
        for sample in samples:
            total += sample.get_utility()

        return total / len(samples)

    @dispatch(ReduceQuery)
    def reduce(self, query):
        """
        Reduces the Bayesian network to a subset of its variables and returns the
        result.

        NB: the equivalent "reduce" method includes additional speed-up methods to
        simplify the reduction process.

        :param query: the reduction query
        :return: the reduced Bayesian network
        """
        network = query.get_network()

        query_vars = query.get_query_vars()
        is_query = LikelihoodWeighting(query, self._nr_samples, self._max_sampling_time)

        samples = is_query.get_samples()

        full_distrib = EmpiricalDistribution(samples)

        reduced_network = BNetwork()
        for variable in query.get_sorted_query_vars():
            input_node_ids = network.get_node(variable).get_ancestor_ids(query_vars)
            for input_node_id in list(input_node_ids):
                input_node = reduced_network.get_chance_node(input_node_id)
                if isinstance(input_node.get_distrib(), ContinuousDistribution):
                    input_node_ids.remove(input_node_id)

            distrib = full_distrib.get_marginal(variable, input_node_ids)

            node = ChanceNode(variable, distrib)
            for input_node_id in input_node_ids:
                node.add_input_node(reduced_network.get_node(input_node_id))
            reduced_network.add_node(node)

        return reduced_network

    @dispatch(BNetwork, Collection, Assignment)
    def reduce(self, network, query_vars, evidence):
        return super().reduce(network, query_vars, evidence)

    @dispatch(BNetwork, Collection)
    def reduce(self, network, query_vars):
        return super().reduce(network, query_vars)

    @dispatch(Query, Callable)
    def get_weighted_samples(self, query, weight_scheme):
        """
        Returns an empirical distribution for the particular query, after reweighting
        each samples based on the provided weighting scheme.

        :param query: the query
        :param weightScheme: the weighting scheme for the samples
        :return: the resulting empirical distribution for the query variables, after
                 reweigthing
        """
        weighted_queries = dict()
        weighted_queries[query] = weight_scheme

        self.get_weighted_samples(weighted_queries)

    @dispatch(dict)
    def get_weighted_samples(self, weighted_queries):
        """
        Returns an empirical distribution for the particular query, after reweighting
        each samples based on the provided weighting scheme.

        :param weighted_queries: the weighting queries
        :return: the resulting empirical distribution for the query variables, after
                 reweigthing
        """
        distrib = EmpiricalDistribution()

        for query in weighted_queries.keys():
            weight_scheme = weighted_queries[query]
            is_query = LikelihoodWeighting(query, self._nr_samples, self._max_sampling_time)

            samples = is_query.get_samples()
            weight_scheme(samples)

            intervals = Intervals(samples, lambda x: x.get_weight())
            for _ in range(len(samples)):
                distrib.add_sample(intervals.sample())

        return distrib
