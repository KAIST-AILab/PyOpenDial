import logging
import math
from collections import OrderedDict, Collection

from multipledispatch import dispatch

from bn.b_network import BNetwork
from bn.distribs.distribution_builder import ConditionalTableBuilder as ConditionalTableBuilder, MultivariateTableBuilder as MultivariateTableBuilder
from bn.distribs.utility_table import UtilityTable
from bn.nodes.chance_node import ChanceNode
from datastructs.assignment import Assignment
from inference.inference_algorithm import InferenceAlgorithm
from inference.query import ProbQuery, UtilQuery, ReduceQuery
from utils.inference_utils import InferenceUtils

dispatch_namespace = dict()


class NaiveInference(InferenceAlgorithm):
    """
    Algorithm for naive probabilistic inference, based on computing the full joint
    distribution, and then summing everything.
    """
    log = logging.getLogger('PyOpenDial')

    def __init__(self):
        super(NaiveInference, self).__init__()

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
        Queries the probability distribution encoded in the Bayesian Network, given a
        set of query variables, and some evidence.

        :param query: the full query result
        :return: the corresponding multivariate table failed
        """
        network = query.get_network()
        query_vars = set(query.get_query_vars())
        evidence = query.get_evidence()

        full_joint = NaiveInference.get_full_joint(network, False)

        # TODO: Do I have to use TreeMap(Java) like data-structure? ... OrderedDict is not same with TreeMap
        query_values = OrderedDict()
        for node in network.get_chance_nodes():
            if node.get_id() in query_vars:
                query_values[node.get_id()] = node.get_values()

        query_assignments = InferenceUtils.get_all_combinations(query_values)

        query_result = MultivariateTableBuilder()

        for query_assignment in query_assignments:
            sum = 0.
            for assignment in full_joint.keys():
                if assignment.contains(query_assignment) and assignment.contains(evidence):
                    sum += full_joint[assignment]

            query_result.add_row(query_assignment, sum)

        query_result.normalize()
        return query_result.build()

    @staticmethod
    @dispatch(BNetwork, bool, namespace=dispatch_namespace)
    def get_full_joint(network, include_actions):
        """
        Computes the full joint probability distribution for the Bayesian Network

        :param network: the Bayesian network
        :param include_actions: whether to include action nodes or not
        :return: the resulting joint distribution
        """
        # TODO: Do I have to use TreeMap(Java) like data-structure? ... OrderedDict is not same with TreeMap
        all_values = OrderedDict()
        for chance_node in network.get_chance_nodes():
            all_values[chance_node.get_id()] = chance_node.get_values()

        if include_actions:
            for action_node in network.get_action_nodes():
                all_values[action_node.get_id()] = action_node.get_values()

        full_assignments = InferenceUtils.get_all_combinations(all_values)
        result = dict()
        for assignment in full_assignments:
            joint_log_prob = 0.
            for chance_node in network.get_chance_nodes():
                trimmed_condition = assignment.get_trimmed(chance_node.get_input_node_ids())
                joint_log_prob += math.log10(chance_node.get_prob(trimmed_condition, assignment.get_value(chance_node.get_id())))

            if include_actions:
                for action_node in network.get_action_nodes():
                    joint_log_prob += math.log10(action_node.get_prob(assignment.get_value(action_node.get_id())))

            result[assignment] = pow(10, joint_log_prob)

        return result

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
        Computes the utility distribution for the Bayesian network, depending on the
        value of the action variables given as parameters.

        :param query: the full query
        :return: the corresponding utility table
        """
        network = query.get_network()
        query_vars = set(query.get_query_vars())
        evidence = query.get_evidence()

        full_joint = NaiveInference.get_full_joint(network, True)

        # TODO: Do I have to use TreeMap(Java) like data-structure? ... OrderedDict is not same with TreeMap
        action_values = OrderedDict()
        for node in network.get_nodes():
            if node.get_id() in query_vars:
                action_values[node.get_id()] = node.get_values()

        action_assignments = InferenceUtils.get_all_combinations(action_values)
        table = UtilityTable()

        for action_assignment in action_assignments:
            total_utility = 0.
            total_prob = 0.

            for joint_assignment in full_joint.keys():
                if not joint_assignment.contains(evidence):
                    continue

                total_utility_for_assignment = 0.
                state_and_action_assignment = Assignment([joint_assignment, action_assignment])

                for value_node in network.get_utility_nodes():
                    utility = value_node.get_utility(state_and_action_assignment)
                    total_utility_for_assignment += utility

                total_utility += (total_utility_for_assignment * full_joint.get(joint_assignment))
                total_prob += full_joint.get(joint_assignment)

            table.set_util(action_assignment, total_utility / total_prob)

        return table

    @dispatch(ReduceQuery)
    def reduce(self, query):
        """
        Reduces the Bayesian network to a subset of its variables. This reduction
        operates here by generating the possible conditional assignments for every
        retained variables, and calculating the distribution for each assignment.

        :param query: the reduction query
        :return: the reduced network
        """
        network = query.get_network()
        query_vars = set(query.get_query_vars())
        evidence = query.get_evidence()

        original_sorted_node_ids = network.get_sorted_node_ids()
        sorted_node_ids = list()
        for node_id in original_sorted_node_ids:
            if node_id in query_vars:
                sorted_node_ids.append(node_id)

        sorted_node_ids = list(reversed(sorted_node_ids))
        reduced_network = BNetwork()
        for variable_id in sorted_node_ids:
            direct_ancestors = network.get_node(variable_id).get_ancestor_ids(query_vars)

            input_values = dict()
            for direct_ancestor in direct_ancestors:
                input_values[direct_ancestor] = network.get_node(variable_id).get_values()

            assignments = InferenceUtils.get_all_combinations(input_values)

            builder = ConditionalTableBuilder(variable_id)
            for assignment in assignments:
                new_evidence = Assignment([evidence, assignment])
                result = self.query_prob(network, variable_id, new_evidence)
                builder.add_rows(assignment, result.get_table())

            chance_node = ChanceNode(variable_id, builder.build())
            for ancestor in direct_ancestors:
                chance_node.add_input_node(reduced_network.get_node(ancestor))

            reduced_network.add_node(chance_node)

        return reduced_network

    @dispatch(BNetwork, Collection, Assignment)
    def reduce(self, network, query_vars, evidence):
        return super().reduce(network, query_vars, evidence)

    @dispatch(BNetwork, Collection)
    def reduce(self, network, query_vars):
        return super().reduce(network, query_vars)
