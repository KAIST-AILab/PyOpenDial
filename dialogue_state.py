import copy
import logging
import threading
from collections import Collection
from xml.etree.ElementTree import Element, ElementTree

from multipledispatch import dispatch

from bn.b_network import BNetwork
from bn.distribs.categorical_table import CategoricalTable
from bn.distribs.independent_distribution import IndependentDistribution
from bn.distribs.multivariate_distribution import MultivariateDistribution
from bn.distribs.prob_distribution import ProbDistribution
from bn.distribs.single_value_distribution import SingleValueDistribution
from bn.nodes.action_node import ActionNode
from bn.nodes.chance_node import ChanceNode
from bn.nodes.utility_node import UtilityNode
from bn.values.value_factory import ValueFactory
from datastructs.assignment import Assignment
from datastructs.value_range import ValueRange
from domains.rules.distribs.anchored_rule import AnchoredRule
from domains.rules.distribs.equivalence_distribution import EquivalenceDistribution
from domains.rules.distribs.output_distribution import OutputDistribution
from domains.rules.rule import Rule, RuleType
from inference.approximate.sampling_algorithm import SamplingAlgorithm
from inference.switching_algorithm import SwitchingAlgorithm


class DialogueStateWrapper(BNetwork):
    pass


class DialogueState(DialogueStateWrapper):
    """
    Representation of a dialogue state. A dialogue state is essentially a directed
    graphical model (i.e. a Bayesian or decision network) over a set of specific state
    variables. Probabilistic rules can be applied on this dialogue state in order to
    update its content. After applying the rules, the dialogue state is usually pruned
    to only retain relevant state variables.

    The dialogue state may also include an assignment of evidence values. A subset of
    state variables can be marked as denoting parameter variables.
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    # ===============================
    # DIALOGUE STATE CONSTRUCTION
    # ===============================

    def __init__(self, arg1=None, arg2=None):
        if arg1 is None and arg2 is None:
            """
            Creates a new, empty dialogue state.
            """
            super().__init__()
            super().reset(BNetwork())
            self._evidence = Assignment()  # evidence values for state variables
            self._parameter_vars = set()  # Subset of variables that denote parameters
            self._incremental_vars = set()  # Subset of variables that are currently incrementally constructed

            self._init_lock()
        elif isinstance(arg1, BNetwork) and arg2 is None:
            network = arg1
            """
            Creates a new dialogue state that contains the Bayesian network provided as
            argument.

            :param network: the Bayesian network to include
            """
            super().__init__()
            super().reset(network)
            self._evidence = Assignment()  # evidence values for state variables
            self._parameter_vars = set()  # Subset of variables that denote parameters
            self._incremental_vars = set()  # Subset of variables that are currently incrementally constructed

            self._init_lock()
        elif isinstance(arg1, Collection) and isinstance(arg2, Assignment):
            nodes = arg1
            evidence = arg2
            """
            Creates a new dialogue state that contains the set of nodes provided as
            argument.

            :param nodes: the nodes to include
            :param evidence: the evidence
            """
            super().__init__(nodes)
            self._evidence = Assignment(evidence)
            self._parameter_vars = set()  # Subset of variables that denote parameters
            self._incremental_vars = set()  # Subset of variables that are currently incrementally constructed

            self._init_lock()
        elif isinstance(arg1, BNetwork) and isinstance(arg2, Assignment):
            network = arg1
            evidence = arg2
            """
            Creates a new dialogue state that contains the Bayesian network provided as
            argument.

            :param network: the Bayesian network to include
            :param evidence: the additional evidence
            """
            super().__init__()
            super().reset(network)
            self._evidence = Assignment(evidence)
            self._parameter_vars = set()  # Subset of variables that denote parameters
            self._incremental_vars = set()  # Subset of variables that are currently incrementally constructed

            self._init_lock()
        else:
            raise NotImplementedError("UNDEFINED PARAMETERS")

    def _init_lock(self):
        # TODO: need refactoring (decorator?)
        self._locks = {
            'add_to_state_assignment': threading.RLock(),
            'add_to_state_mv_distn': threading.RLock(),
            'add_to_state_incremental': threading.RLock(),
            'add_to_state_dialogue_state': threading.RLock(),
            'add_to_state_bnetwork': threading.RLock(),
            'remove_from_state': threading.RLock(),
            'get_new_variables': threading.RLock(),
            'get_new_action_variables': threading.RLock(),
        }

    @dispatch(BNetwork)
    def reset(self, network):
        """
        Resets the content of the dialogue state to the network contained as argument
        (and deletes the rest).

        :param network: the Bayesian network
        """
        if self == network:
            return

        self._evidence.remove_pairs(self.get_chance_node_ids())
        super(DialogueState, self).reset(network)

        if isinstance(network, DialogueState):
            self._evidence.add_assignment(network.get_evidence())

    @dispatch(Collection)
    def clear_evidence(self, variables):
        """
        Clear the assignment of values for the variables provided as argument

        :param variables: the variables for which to clear the assignment
        """
        self._evidence.remove_pairs(variables)

    @dispatch(Assignment)
    def add_evidence(self, assignment):
        """
        Adds a new assignment of values to the evidence

        :param assignment: the assignment of values to add
        """
        self._evidence.add_assignment(assignment)

    @dispatch(BNetwork)
    def set_parameters(self, parameters):
        """
        Adds a set of parameter variables to the dialogue state

        :param parameters: the parameters
        """
        self.add_network(parameters)
        self._parameter_vars.clear()
        self._parameter_vars.update(parameters.get_chance_node_ids())

    # ===============================
    # STATE UPDATE
    # ===============================

    @dispatch(Assignment)
    def add_to_state(self, assignment):
        """
        Adds the content provided as argument to the dialogue state. If the state
        variables in the assignment already exist, they are erased.

        :param assignment: the value assignment to add
        """
        with self._locks['add_to_state_assignment']:
            for variable in assignment.get_variables():
                self.add_to_state(SingleValueDistribution(variable, assignment.get_value(variable)))

    @dispatch(MultivariateDistribution)
    def add_to_state(self, distrib):
        """
        Adds the content provided as argument to the dialogue state. If the state
        variables in the assignment already exist, they are erased.

        :param distrib: the multivariate distribution to add be added.
        :return:
        """
        with self._locks['add_to_state_mv_distn']:
            for variable in distrib.get_variables():
                self.add_to_state(distrib.get_marginal(variable))

    @dispatch(ProbDistribution)
    def add_to_state(self, distrib):
        """
        Adds a new node to the dialogue state with the distribution provided as
        argument.

        :param distrib: the distribution to include
        """
        variable = distrib.get_variable() + "'"
        self.set_as_committed(variable)
        distrib.modify_variable_id(distrib.get_variable(), variable)
        chance_node = ChanceNode(variable, distrib)

        if self.has_node(variable):
            to_remove = self.get_node(variable)
            self.remove_nodes(to_remove.get_descendant_ids())
            self.remove_node(to_remove.get_id())

        for input_variable in distrib.get_input_variables():
            if self.has_chance_node(input_variable):
                chance_node.add_input_node(self.get_chance_node(input_variable))

        self.add_node(chance_node)
        self._connect_to_predictions(chance_node)
        if variable in self._incremental_vars:
            self._incremental_vars.remove(variable)

    @dispatch(CategoricalTable, bool)
    def add_to_state_incremental(self, distrib, follow_previous):
        """
        Concatenates the current value for the new content onto the current content,
        if followPrevious is true. Else, simply overwrites the current content of the variable.

        :param distrib: the distribution to add as incremental unit of content
        :param follow_previous: whether the results should be concatenated to the previous values,
                                or reset the content (e.g. when starting a new utterance
        """
        with self._locks['add_to_state_incremental']:
            if follow_previous is None:
                raise ValueError()

            if not follow_previous:
                self.set_as_committed(distrib.get_variable())

            variable = distrib.get_variable()
            if self.has_chance_node(variable) and self.is_incremental(variable) and follow_previous:
                new_table = self.query_prob(variable).to_discrete().concatenate(distrib)
                self.get_chance_node(variable).set_distrib(new_table)
                self.get_chance_node(variable).set_id(variable + "'")
            else:
                self.add_to_state(distrib)

            self._incremental_vars.add(variable)

    @dispatch(DialogueStateWrapper)
    def add_to_state(self, new_state):
        """
        Merges the dialogue state included as argument into the current one.

        :param new_state: the state to merge into the current state dialogue state could not be merged
        """
        with self._locks['add_to_state_dialogue_state']:
            self.add_to_state(new_state)
            self._evidence.add_assignment(new_state.get_evidence().add_primes())

    @dispatch(BNetwork)
    def add_to_state(self, new_state):
        """
        Merges the dialogue state included as argument into the current one.

        :param new_state: the state to merge into the current state dialogue state could not be merged
        """
        with self._locks['add_to_state_bnetwork']:
            for chance_node in new_state.get_chance_nodes():
                chance_node.set_id(chance_node.get_id() + "'")
                self.add_node(chance_node)
                self._connect_to_predictions(chance_node)

    @dispatch(str)
    def remove_from_state(self, variable_id):
        """
        Removes the variable from the dialogue state

        :param variable_id: the node to remove
        """
        with self._locks['remove_from_state']:
            self.add_to_state(Assignment(variable_id, ValueFactory.none()))

    @dispatch(Rule)
    def apply_rule(self, rule):
        """
        Applies a (probability or utility) rule to the dialogue state:
        - For a probability rule, the method creates a chance node containing the rule effects depending on the input
          variables of the rule, and connected via outgoing edges to the output variables.
        - For a utility rule, the method creates a utility node specifying the utility of particular actions specified
          by the rule depending on the input variables.

        The method creates both the rule node, its corresponding output or action
        nodes, and the directed edges resulting from the rule application. See Pierre
        Lison's PhD thesis, Section 4.3 for details.

        :param rule: the rule to apply
        """
        slots = self.get_matching_slots(rule.get_input_variables()).linearize()
        for filled_slot in slots:
            anchored_rule = AnchoredRule(rule, self, filled_slot)
            if anchored_rule.is_relevant():
                if rule.get_rule_type() == RuleType.PROB:
                    self._add_probability_rule(anchored_rule)
                elif rule.get_rule_type() == RuleType.UTIL:
                    self._add_utility_rule(anchored_rule)

    @dispatch()
    def set_as_new(self):
        """
        Sets the dialogue state to consist of all new variables (to trigger right after the system initialisation).
        """
        for chance_node in self.get_chance_nodes():
            chance_node.set_id(chance_node.get_id() + "'")

    # ===============================
    # GETTERS
    # ===============================

    @dispatch()
    def get_evidence(self):
        """
        Returns the evidence associated with the dialogue state.
        :return: the assignment of values for the evidence
        """
        return Assignment(self._evidence)

    @dispatch(str)
    def query_prob(self, variable):
        """
        Returns the probability distribution corresponding to the values of the state
        variable provided as argument.

        :param variable: the variable label to query
        :return: the corresponding probability distribution
        """
        return self.query_prob(variable, True)

    @dispatch(str, bool)
    def query_prob(self, variable, include_evidence):
        """
        Returns the probability distribution corresponding to the values of the state
        variable provided as argument.

        :param variable: the variable label to query
        :param include_evidence: whether to include or ignore the evidence in the dialogue state
        :return: the corresponding probability distribution
        """
        if self.has_chance_node(variable):
            chance_node = self.get_chance_node(variable)

            if isinstance(chance_node.get_distrib(), IndependentDistribution) and chance_node.get_clique().isdisjoint(
                    self._evidence.get_variables()):
                return chance_node.get_distrib()
            else:
                try:
                    query_evidence = self._evidence if include_evidence else Assignment()
                    return SwitchingAlgorithm().query_prob(self, variable, query_evidence)
                except Exception as e:
                    self.log.warning("Error querying variable %s : %s" % (variable, e))
                    raise ValueError()
        else:
            self.log.warning("Variable %s not included in the dialogue state" % variable)
            raise ValueError()

    @dispatch(Collection)  # collection of strings
    def query_prob(self, variables):
        """
        Returns the probability distribution corresponding to the values of the state
        variables provided as argument.

        :param variables: the variable labels to query
        :return: the corresponding probability distribution
        """
        if not set(variables).issubset(self.get_node_ids()):
            self.log.warning(variables + " not contained in " + self.get_node_ids())
            raise ValueError()
        try:
            return SwitchingAlgorithm().query_prob(self, variables, self._evidence)
        except Exception as e:
            self.log.warning("cannot perform inference: %s" % e)
            raise ValueError()

    @dispatch(Collection)  # collection of strings
    def query_util(self, variables):
        """
        Returns the utility table associated with a particular set of (state or
        action) variables.

        :param variables: the state or action variables to consider
        :return: the corresponding utility table
        """
        try:
            return SwitchingAlgorithm().query_util(self, variables, self._evidence)
        except Exception as e:
            self.log.warning("cannot perform inference: " + e)
            raise ValueError()

    @dispatch()
    def query_util(self):
        """
        Returns the total utility of the dialogue state (marginalising over all
        possible state variables).

        :return: the total utility
        """
        try:
            return SamplingAlgorithm().query_util(self)
        except Exception as e:
            self.log.warning("cannot perform inference: " + e)
            raise ValueError()

    @dispatch(Collection)  # collection of Template
    def get_matching_slots(self, templates):
        """
        Returns the possible filled values for the underspecified slots in the templates, on the basis of the variables
        in the dialogue sate.

        :param templates: the templates to apply
        :return: the possible values (as a value range)
        """
        value_range = ValueRange()
        for template in templates:
            if not template.is_under_specified():
                continue

            for node_id in self.get_chance_node_ids():
                if node_id.endswith("'"):
                    continue

                match = template.match(node_id)
                if not match.is_matching():
                    continue

                value_range.add_assign(match)

        return value_range

    @dispatch()
    def get_parameter_ids(self):
        """
        Returns the set of parameter variables in the dialogue state

        :return: the parameter variables
        """
        return self._parameter_vars

    @dispatch()
    def get_sample(self):
        """
        Returns a sample of all the variables in the dialogue state

        :return: a sample assignment
        """
        return SamplingAlgorithm.extract_sample(self, self.get_chance_node_ids())

    @dispatch()
    def get_new_variables(self):
        """
        Returns the set of updated variables in the dialogue state (that is, the ones that have
        a prime ' in their label).

        :return: the list of updated variables
        """
        with self._locks['get_new_variables']:
            new_variables = set()
            for variable in self.get_chance_node_ids():
                if variable[-1] == "'":
                    new_variables.add(variable[:-1])
            return new_variables

    @dispatch()
    def get_new_action_variables(self):
        """
        Returns the set of new action variables in the dialogue state (that is, the ones that have
        a prime ' in their label).

        :return: the list of new action variables
        """
        with self._locks['get_new_action_variables']:
            new_variables = set()
            for variable in self.get_action_node_ids():
                if variable[-1] == "'":
                    new_variables.add(variable[:-1])
            return new_variables

    @dispatch(str)
    def is_incremental(self, variable):
        """
        Returns true if the given variable is in incremental mode, false otherwise

        :param variable: the variable label (str)
        :return: true if the variable is incremental, false otherwise
        """
        return (variable.replace("'", "")) in self._incremental_vars

    @dispatch()
    def get_incremental_vars(self):
        """
        Returns the set of state variables that are in incremental mode

        :return: the set of incremental variables
        """
        return self._incremental_vars

    # ===============================
    # STATE UPDATE
    # ===============================

    @dispatch(str)
    def set_as_committed(self, variable):
        """
        setAsCommitted(variable)

        :param variable: variable (str)
        """
        if variable in self._incremental_vars:
            self._incremental_vars.remove(variable)
            from modules.state_pruner import StatePruner
            StatePruner.prune(self)

    @dispatch()
    def reduce(self):
        """
        Prunes the dialogue state (see Section 4.4 of Pierre Lison's PhD thesis).
        """
        if len(self.get_new_variables()) > 0 or not self._evidence.is_empty():
            from modules.state_pruner import StatePruner
            StatePruner.prune(self)

    def __copy__(self):
        """
        Returns a copy of the dialogue state

        :return: the copy
        """
        dialogue_state = DialogueState(super().__copy__())
        dialogue_state.add_evidence(copy.copy(self._evidence))
        dialogue_state._parameter_vars = set(self._parameter_vars)
        dialogue_state._incremental_vars = set(self._incremental_vars)
        return dialogue_state

    def __str__(self):
        """
        Returns a string representation of the dialogue state

        :return: the string representation
        """
        string_val = super(DialogueState, self).__str__()
        if not self._evidence.is_empty():
            string_val += '[evidence=%s]' % str(self._evidence)
        return string_val

    def __hash__(self):
        """
        Returns the hashcode for the dialogue state

        :return: the hashcode
        """
        return super(DialogueState, self).__hash__() - 2 * hash(self._evidence)

    @dispatch(Collection)
    def generate_xml(self, vars_to_record):
        """
        Generates an XML element that encodes the dialogue state content.

        :param doc: the document to which the element must comply
        :param vars_to_record: the set of variables to record
        :return: the resulting XML element
        """
        root = Element("state")
        # root = Element("")
        for node_id in vars_to_record:
            if node_id in self.get_chance_node_ids():
                distrib = self.query_prob(node_id)
                var = distrib.generate_xml()
                root.append(var)
        return root

    # ===============================
    # PRIVATE METHODS
    # ===============================

    @dispatch(AnchoredRule)
    def _add_probability_rule(self, rule):
        """
        Adds the probability rule to the dialogue state

        :param rule: the anchored rule (must be of type PROB) nodes fails
        """
        rule_id = rule.get_variable()
        if self.has_chance_node(rule_id):
            self.remove_node(rule_id)

        rule_node = ChanceNode(rule_id, rule)
        rule_node.get_values()

        for var in rule.get_input_variables().union(rule.get_parameters()):
            if self.has_chance_node(var):
                rule_node.add_input_node(self.get_chance_node(var))
            else:
                raise ValueError('undefined node type of %s' % var)
        self.add_node(rule_node)

        for updated_var in rule.get_outputs():
            if not self.has_node(updated_var):
                output_distrib = OutputDistribution(updated_var)
                output_node = ChanceNode(updated_var, output_distrib)
                self.add_node(output_node)
                self._connect_to_predictions(output_node)
            else:
                output_node = self.get_chance_node(updated_var)
                output_distrib = output_node.get_distrib()

            output_node.add_input_node(rule_node)
            output_distrib.add_anchored_rule(rule)

    @dispatch(AnchoredRule)
    def _add_utility_rule(self, rule):
        """
        Adds the utility rule to the dialogue state.

        :param rule: the anchored rule (must be of type UTIL) nodes fails
        """
        rule_id = rule.get_variable()
        if self.has_utility_node(rule_id):
            self.remove_node(rule_id)

        rule_node = UtilityNode(rule_id)
        rule_node.set_distrib(rule)
        for var in rule.get_input_variables():
            rule_node.add_input_node(self.get_chance_node(var))
        for param in rule.get_parameters():
            rule_node.add_input_node(self.get_chance_node(param))
        self.add_node(rule_node)

        actions = rule.get_output_range()
        for action_var in actions.get_variables():
            if not self.has_action_node(action_var):
                action_node = ActionNode(action_var)
                self.add_node(action_node)
            else:
                action_node = self.get_action_node(action_var)

            rule_node.add_input_node(action_node)
            action_node.add_values(actions.get_values(action_var))

    @dispatch(ChanceNode)
    def _connect_to_predictions(self, output_node):
        """
        Connects the chance node to its prior predictions (if any)

        :param output_node: the output node to connect
        """
        output_var = output_node.get_id()

        base_var = output_var[:-1]
        predict_equiv = base_var + "^p"
        if self.has_chance_node(predict_equiv) and "^p" not in output_var:
            equality_node = ChanceNode("=_" + base_var, EquivalenceDistribution(base_var))
            equality_node.add_input_node(output_node)
            equality_node.add_input_node(self.get_node(predict_equiv))
            self.add_evidence(Assignment(equality_node.get_id(), True))
            self.add_node(equality_node)
