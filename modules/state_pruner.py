import copy
import logging
from collections import Collection

from multipledispatch import dispatch

from bn.b_network import BNetwork
from bn.distribs.categorical_table import CategoricalTable
from bn.distribs.marginal_distribution import MarginalDistribution
from bn.nodes.chance_node import ChanceNode
from bn.values.value_factory import ValueFactory
from datastructs.assignment import Assignment
from dialogue_state import DialogueState
from domains.rules.distribs.anchored_rule import AnchoredRule
from domains.rules.distribs.equivalence_distribution import EquivalenceDistribution
from inference.switching_algorithm import SwitchingAlgorithm

dispatch_namespace = dict()


class StatePruner:
    """
    Prunes the dialogue state by removing all intermediary nodes (that is, rule nodes,
    utility and action nodes, equivalence nodes, and outdated versions of updated
    variables).
    """
    log = logging.getLogger('PyOpenDial')

    value_pruning_threshold = .01
    enable_reduction = True

    @staticmethod
    @dispatch(DialogueState, namespace=dispatch_namespace)
    def prune(state):
        """
        Prunes the state of all the non-necessary nodes. the operation selects a
        subset of relevant nodes to keep, prunes the irrelevant ones, remove the
        primes from the variable labels, and delete all empty nodes.

        :param state: the state to prune
        """
        # step 1: selection of nodes to keep
        nodes_to_keep = StatePruner.get_nodes_to_keep(state)
        if len(nodes_to_keep) > 0:
            # step 2: reduction
            reduced = StatePruner.reduce(state, nodes_to_keep)
            # step 3: reinsert action and utility nodes (if necessary)
            StatePruner.reinsert_action_and_utility_nodes(reduced, state)
            # step 4: remove the primes from the identifiers
            StatePruner.remove_primes(reduced)
            # step 5: filter the distribution and remove and empty nodes
            StatePruner.remove_spurious_nodes(reduced)
            # step 6: and final reset the state to the reduced form
            state.reset(reduced)
        else:
            state.reset(BNetwork())

    @staticmethod
    @dispatch(DialogueState, namespace=dispatch_namespace)
    def get_nodes_to_keep(state):
        """
        Selects the set of variables to retain in the dialogue state.

        :param state: the dialogue state
        :return: the set of variable labels to keep
        """
        nodes_to_keep = set()

        for chance_node in state.get_chance_nodes():
            if chance_node.get_id()[:2] == "=_" or chance_node.get_id()[-2:] == "^t" or chance_node.get_id()[-2:] == "^o":
                continue
            elif StatePruner.enable_reduction and isinstance(chance_node.get_distrib(), AnchoredRule):
                continue
            elif len(chance_node.get_input_node_ids()) < 3 and chance_node.get_nb_values() == 1 and list(chance_node.get_values())[0] == ValueFactory.none():
                continue
            elif chance_node.get_id()[-2:] == "^p":
                flag = False
                for node_ids in chance_node.get_output_node_ids():
                    if node_ids[:2] == "=_":
                        flag = True
                        break
                if flag:
                    continue

            if not state.has_chance_node(chance_node.get_id() + "'"):
                nodes_to_keep.add(chance_node.get_id())

            if state.is_incremental(chance_node.get_id()):
                for descendant_id in chance_node.get_descendant_ids():
                    if not state.has_chance_node(descendant_id):
                        continue
                    if state.has_chance_node(descendant_id + "'"):
                        continue
                    nodes_to_keep.add(descendant_id)

            if chance_node.get_id() in state.get_parameter_ids() and not chance_node.has_descendant(state.get_evidence().get_variables()):
                for output_node in chance_node.get_output_nodes(ChanceNode):
                    if not isinstance(output_node.get_distrib(), AnchoredRule):
                        continue
                    nodes_to_keep.add(output_node.get_id())

        return nodes_to_keep

    @staticmethod
    @dispatch(DialogueState, set, namespace=dispatch_namespace)
    def reduce(state, nodes_to_keep):
        """
        Reduces a Bayesian network to a subset of variables. The method is divided in
        three steps:

        - The method first checks whether inference is necessary at all or whether
        the current network can be returned as it is.
        - If inference is necessary, the algorithm divides the network into cliques
        and performs inference on each clique separately.
        - Finally, if only one clique is present, the reduction selects the best
        algorithm and return the result of the reduction process.

        :param state: the dialogue state to reduce
        :param nodes_to_keep: the nodes to preserve in the reduction

        :return: the reduced dialogue state
        """
        evidence = state.get_evidence()
        if evidence.contains_vars(nodes_to_keep):
            # if all nodes to keep are included in the evidence, no inference is needed
            new_state = DialogueState()
            for node_to_keep in nodes_to_keep:
                new_node = ChanceNode(node_to_keep, evidence.get_value(node_to_keep))
                new_state.add_node(new_node)

            return new_state
        elif (state.get_node_ids()).issubset(nodes_to_keep):
            # if the current network can be returned as such, do it
            return state
        elif state.is_clique(nodes_to_keep) and not evidence.contains_one_var(nodes_to_keep):
            # if all nodes belong to a single clique and the evidence does not
            # pertain to them, return the subset of nodes
            return DialogueState(state.get_nodes(nodes_to_keep), evidence)
        elif state.contains_distrib(nodes_to_keep, AnchoredRule):
            # if some rule nodes are included
            return StatePruner.reduce_light(state, nodes_to_keep)

        # if the network can be divided into cliques, extract the cliques
        # and do a separate reduction for each
        cliques = state.get_cliques(nodes_to_keep)
        if len(cliques) > 1:
            full_state = DialogueState()
            for clique in cliques:
                clique.intersection_update(nodes_to_keep)
                clique_state = StatePruner.reduce(state, clique)
                full_state.add_network(clique_state)
                full_state.add_evidence(clique_state.get_evidence())

            return full_state

        result = SwitchingAlgorithm().reduce(state, nodes_to_keep, evidence)
        return DialogueState(result)

    @staticmethod
    @dispatch(DialogueState, Collection, namespace=dispatch_namespace)
    def reduce_light(state, nodes_to_keep):
        """
        "lightweight" reduction of the dialogue state (without actual inference).

        :param state: the dialogue state
        :param nodes_to_keep: the nodes to keep
        :return: the reduced dialogue state @
        """
        new_state = DialogueState(state, state.get_evidence())
        for chance_node in new_state.get_chance_nodes():
            if chance_node.get_id() not in nodes_to_keep:
                init_distrib = state.query_prob(chance_node.get_id(), False).to_discrete()
                for output_node in chance_node.get_output_nodes(ChanceNode):
                    new_distrib = MarginalDistribution(output_node.get_distrib(), init_distrib)
                    output_node.set_distrib(new_distrib)

                new_state.remove_node(chance_node.get_id())

        return new_state

    @staticmethod
    @dispatch(DialogueState, namespace=dispatch_namespace)
    def remove_primes(reduced):
        """
        Removes the prime characters from the variable labels in the dialogue state.

        :param reduced: the reduced state @
        """
        for chance_node in reduced.get_chance_nodes():
            if reduced.has_chance_node(chance_node.get_id() + "'"):
                reduced.remove_node(chance_node.get_id())

        for node_id in reduced.get_chance_node_ids():
            if "'" in node_id:
                new_id = node_id.replace("'", "")
                if not reduced.has_chance_node(new_id):
                    reduced.get_chance_node(node_id).set_id(new_id)

    @staticmethod
    @dispatch(DialogueState, namespace=dispatch_namespace)
    def remove_spurious_nodes(reduced):
        """
        Removes all non-necessary nodes from the dialogue state.

        :param reduced: the reduced dialogue state
        """
        for chance_node in set(reduced.get_chance_nodes()):
            if len(chance_node.get_input_nodes()) == 0 and len(chance_node.get_output_nodes()) == 0 and isinstance(chance_node.get_distrib(), CategoricalTable) and chance_node.get_prob(ValueFactory.none()) > 0.99:
                reduced.remove_node(chance_node.get_id())
                continue

            if isinstance(chance_node.get_distrib(), EquivalenceDistribution) and len(chance_node.get_input_node_ids()) == 0:
                reduced.remove_node(chance_node.get_id())

            chance_node.prune_values(StatePruner.value_pruning_threshold)

            if chance_node.get_nb_values() == 1 and len(chance_node.get_output_nodes()) > 0 and len(reduced.get_incremental_vars()) == 0:
                assignment = Assignment(chance_node.get_id(), chance_node.sample())
                for output_node in chance_node.get_output_nodes(ChanceNode):
                    if not isinstance(output_node.get_distrib(), AnchoredRule):
                        cur_distrib = output_node.get_distrib()
                        output_node.remove_input_node(chance_node.get_id())
                        if len(output_node.get_input_node_ids()) == 0:
                            output_node.set_distrib(cur_distrib.get_prob_distrib(assignment))
                        else:
                            output_node.set_distrib(cur_distrib.get_posterior(assignment))

    @staticmethod
    @dispatch(BNetwork, BNetwork, namespace=dispatch_namespace)
    def reinsert_action_and_utility_nodes(reduced, original):
        """
        Reinserts the action and utility nodes in the reduced dialogue state.

        :param reduced: the reduced state
        :param original: the original state @
        """
        for action_node in original.get_action_nodes():
            if not reduced.has_action_node(action_node.get_id()):
                reduced.add_node(copy.copy(action_node))

        for utility_node in original.get_utility_nodes():
            if not reduced.has_utility_node(utility_node.get_id()):
                reduced_utility_node = copy.copy(utility_node)
                reduced.add_node(reduced_utility_node)
                for input in utility_node.get_input_node_ids():
                    if reduced.has_node(input):
                        reduced_utility_node.add_input_node(reduced.get_node(input))
                    elif reduced.has_node(input + "'"):  # TODO: check if this is correct
                        reduced_utility_node.add_input_node(reduced.get_node(input + "'"))
