import logging
from collections import Collection
from copy import copy

from multipledispatch import dispatch

from bn.b_network import BNetwork
from bn.distribs.distribution_builder import CategoricalTableBuilder as CategoricalTableBuilder, ConditionalTableBuilder as ConditionalTableBuilder, MultivariateTableBuilder as MultivariateTableBuilder
from bn.distribs.utility_table import UtilityTable
from bn.nodes.action_node import ActionNode
from bn.nodes.b_node import BNode
from bn.nodes.chance_node import ChanceNode
from bn.nodes.utility_node import UtilityNode
from datastructs.assignment import Assignment
from inference.exact.double_factor import DoubleFactor
from inference.inference_algorithm import InferenceAlgorithm
from inference.query import ProbQuery, UtilQuery, Query, ReduceQuery


class VariableElimination(InferenceAlgorithm):
    """
    Implementation of the Variable Elimination algorithm.
    """

    log = logging.getLogger('PyOpenDial')

    def __init__(self):
        super(VariableElimination, self).__init__()

    @dispatch(BNetwork, Collection, Assignment)
    def query_prob(self, network, query_vars, evidence):
        return super(VariableElimination, self).query_prob(network, query_vars, evidence)

    @dispatch(BNetwork, Collection)
    def query_prob(self, network, query_vars):
        return super(VariableElimination, self).query_prob(network, query_vars)

    @dispatch(BNetwork, str, Assignment)
    def query_prob(self, network, query_var, evidence):
        return super(VariableElimination, self).query_prob(network, query_var, evidence)

    @dispatch(BNetwork, str)
    def query_prob(self, network, query_var):
        return super(VariableElimination, self).query_prob(network, query_var, Assignment())

    @dispatch(ProbQuery)
    def query_prob(self, query):
        """
        Queries for the probability distribution of the set of random variables in the
        Bayesian network, given the provided evidence

        :param query: the full query
        :return: the corresponding categorical table failed
        """
        query_factor = self._create_query_factor(query)

        builder = MultivariateTableBuilder()
        builder.add_rows(query_factor.get_prob_table())
        builder.normalize()
        return builder.build()

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
        Queries for the utility of a particular set of (action) variables, given the
        provided evidence

        :param query: the full query
        :return: the utility distribution
        """
        query_factor = self._create_query_factor(query)
        query_factor.normalize()
        return UtilityTable(query_factor.get_util_table())

    @dispatch(Query)
    def _create_query_factor(self, query):
        """
        Generates the full double factor associated with the query variables, using
        the variable-elimination algorithm.

        :param query: the query
        :return: the full double factor containing all query variables occurred during
                 the inference
        """
        factors = list()
        query_vars = query.get_query_vars()
        evidence = query.get_evidence()

        for node in query.get_filtered_sorted_nodes():
            basic_factor = self._make_factor(node, evidence)
            if not basic_factor.is_empty():
                factors.append(basic_factor)

                if node.get_id() not in query_vars:
                    factors = self._sum_out(node.get_id(), factors)

        final_product = self._point_wise_product(factors)
        final_product = self._add_evidence_pairs(final_product, query)
        final_product.trim(query_vars)
        return final_product

    @dispatch(BNode, Assignment)
    def _make_factor(self, node, evidence):
        """
        Creates a new factor given the probability distribution defined in the
        Bayesian node, and the evidence (which needs to be matched)

        :param node: the Bayesian node
        :param evidence: the evidence
        :return: the factor for the node
        """
        factor = DoubleFactor()
        flat_table = node.get_factor()
        for assignment in flat_table.keys():
            if assignment.consistent_with(evidence):
                assignment2 = Assignment(assignment)
                assignment2.remove_pairs(evidence.get_variables())

                if isinstance(node, ChanceNode) or isinstance(node, ActionNode):
                    factor.add_entry(assignment2, flat_table[assignment], 0.)
                elif isinstance(node, UtilityNode):
                    factor.add_entry(assignment2, 1., flat_table[assignment])

        return factor

    @dispatch(str, list)
    def _sum_out(self, node_id, factors):
        """
        * Sums out the variable from the pointwise product of the factors, and returns
        * the result
        *
        :param node_id: the Bayesian node corresponding to the variable
        :param factors: the factors to sum out
        :return: the summed out factor
        """
        dependent_factors = list()
        remaining_factors = list()

        for factor in factors:
            if node_id not in factor.get_variables():
                remaining_factors.append(factor)
            else:
                dependent_factors.append(factor)

        product_dependent_factors = self._point_wise_product(dependent_factors)
        sum_dependent_factors = self._sum_out_dependent(node_id, product_dependent_factors)

        if not sum_dependent_factors.is_empty():
            remaining_factors.append(sum_dependent_factors)

        return remaining_factors

    @dispatch(str, DoubleFactor)
    def _sum_out_dependent(self, node_id, factor):
        """
        Sums out the variable from the given factor, and returns the result

        :param node_id: the Bayesian node corresponding to the variable
        :param factor: the factor to sum out
        :return: the summed out factor
        """
        sum_factor = DoubleFactor()
        for assignment in factor.get_values():
            reduced_assignment = Assignment(assignment)
            reduced_assignment.remove_pair(node_id)
            entry = factor.get_entry(assignment)
            prob = entry[0]
            utility = entry[1]
            sum_factor.increment_entry(reduced_assignment, prob, prob * utility)

        sum_factor.normalize_util()
        return sum_factor

    @dispatch(list)
    def _point_wise_product(self, factors):
        """
        Computes the pointwise matrix product of the list of factors

        :param factors: the factors
        :return: the pointwise product of the factors
        """
        if len(factors) == 0:
            factor = DoubleFactor()
            factor.add_entry(Assignment(), 1., 0.)
            return factor
        elif len(factors) == 1:
            return factors[0]

        factor = factors.pop(0)
        for f in factors:
            temp_factor = DoubleFactor()
            shared_vars = set(f.get_variables()).intersection(factor.get_variables())

            for assignment in f.get_values():
                prob, utility = f.get_entry(assignment)

                for assignment2 in factor.get_values():
                    if assignment2.consistent_with(assignment, shared_vars):
                        prob2, utility2 = factor.get_entry(assignment2)
                        product = prob * prob2
                        sum = utility + utility2

                        temp_factor.add_entry(Assignment([assignment, assignment2]), product, sum)

            factor = temp_factor

        return factor

    @dispatch(DoubleFactor, Query)
    def _add_evidence_pairs(self, factor, query):
        """
        In case of overlap between the query variables and the evidence (this happens
        when a variable specified in the evidence also appears in the query), extends
        the distribution to add the evidence assignment pairs.

        :param factor: the factor
        :param query: the query
        :return: updated factor
        """
        inter = set(query.get_query_vars())
        inter.intersection_update(query.get_evidence().get_variables())
        evidence = query.get_evidence().get_trimmed(inter)

        if len(inter) > 0:
            new_factor = DoubleFactor()
            for assignment in factor.get_assignments():
                assignment2 = Assignment([assignment, evidence])
                prob, utility = factor.get_entry(assignment)
                new_factor.add_entry(assignment2, prob, utility)

            return new_factor

        return factor


    @dispatch(BNetwork, Collection, Assignment)
    def reduce(self, network, query_vars, evidence):
        return super().reduce(network, query_vars, evidence)

    @dispatch(BNetwork, Collection)
    def reduce(self, network, query_vars):
        return super().reduce(network, query_vars)

    @dispatch(ReduceQuery)
    def reduce(self, query):
        """
        Reduces the Bayesian network by retaining only a subset of variables and
        marginalising out the rest.

        :param query: the query containing the network to reduce, the variables to
                      retain, and possible evidence.
        :return: the probability distributions for the retained variables reduction
                 operation failed
        """
        network = query.get_network()
        query_vars = query.get_query_vars()

        query_factor = self._create_query_factor(query)
        reduced_network = BNetwork()

        original_sorted_node_ids = network.get_sorted_node_ids()
        sorted_node_ids = list()
        for node_id in original_sorted_node_ids:
            if node_id in query_vars:
                sorted_node_ids.append(node_id)
        sorted_node_ids = list(reversed(sorted_node_ids))

        for variable in sorted_node_ids:
            direct_ancestors = network.get_node(variable).get_ancestor_ids(query_vars)
            factor = self._get_relevant_factor(query_factor, variable, direct_ancestors)
            distrib = self._create_prob_distribution(variable, factor)

            chance_node = ChanceNode(variable, distrib)
            for ancestor in direct_ancestors:
                chance_node.add_input_node(reduced_network.get_node(ancestor))
            reduced_network.add_node(chance_node)

        return reduced_network

    @dispatch(DoubleFactor, str, set)
    def _get_relevant_factor(self, full_factor, head_var, input_vars):
        """
        Returns the factor associated with the probability/utility distribution for
        the given node in the Bayesian network. If the factor encode more than the
        needed distribution, the surplus variables are summed out.

        :param full_factor: the collection of factors in which to search
        :param head_var: the head vars
        :param input_vars: the variable to estimate
        :return: the relevant factor associated with the node could be found
        """
        factor = copy(full_factor)
        for other_var in factor.get_variables():
            if other_var != head_var and other_var not in input_vars:
                summed_out = self._sum_out(other_var, [factor])

                if len(summed_out) > 0:
                    factor = summed_out[0]

        return factor

    @dispatch(str, DoubleFactor)
    def _create_prob_distribution(self, head_var, factor):
        """
        Creates the probability distribution for the given variable, as described by
        the factor. The distribution is normalised, and encoded as a table.

        :param head_var: the variable
        :param factor: the double factor
        :return: the resulting probability distribution
        """
        variables = factor.get_variables()

        if len(variables) == 1:
            factor.normalize()
            builder = CategoricalTableBuilder(head_var)
            for assignment in factor.get_assignments():
                builder.add_row(assignment.get_value(head_var), factor.get_prob_entry(assignment))

            return builder.build()
        else:
            dependent_variables = set(variables)
            if head_var in dependent_variables:
                dependent_variables.remove(head_var)

            factor.normalize(dependent_variables)
            builder = ConditionalTableBuilder(head_var)
            for assignment in factor.get_assignments():
                condition = assignment.get_trimmed(dependent_variables)
                builder.add_row(condition, assignment.get_value(head_var), factor.get_prob_entry(assignment))

            return builder.build()
