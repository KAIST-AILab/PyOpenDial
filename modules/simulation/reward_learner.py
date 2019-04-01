from copy import copy
from math import *

from dialogue_state import DialogueState
from inference.approximate.sampling_algorithm import *
from modules.module import Module


class RewardLearner(Module):
    """
    Module employed during simulated dialogues to automatically estimate a utility
    model from rewards produced by the simulator.
    """

    log = logging.getLogger('PyOpenDial')

    def __init__(self, system):
        from dialogue_system import DialogueSystem
        if not isinstance(system, DialogueSystem):
            raise NotImplementedError("UNDEFINED PARAMETERS")
        """
        Creates the reward learner for the dialogue system.

        :param system: the dialogue system.
        """
        self.system = system
        self.previous_states = {}
        self.sampler = SamplingAlgorithm()

    def start(self):
        """
        Does nothing.
        """
        pass

    def pause(self, should_be_paused):
        """
        Does nothing.
        """
        pass

    def is_running(self):
        """
        Returns true
        """
        return True

    @dispatch(DialogueState, Collection)
    def trigger(self, state, updated_vars):
        """
        * Triggers the reward learner. The module is only triggered whenever a variable
        * of the form R(assignment of action values) is included in the dialogue state
        * by the simulator. In such case, the module checks whether a past dialogue
        * state contains a decision for these action variables, and if yes, update their
        * parameters to reflect the actual received reward.
        *
        * @param state the dialogue state
        * @param updatedVars the list of recently updated variables.
        """
        for evidence_var in state.get_evidence().get_variables():
            if evidence_var.startswith('R(') and evidence_var.endswith(')'):
                actual_action = Assignment.create_from_string(evidence_var[2:])
                actual_utility = state.get_evidence.get_value(evidence_var).get_double()

                if actual_action.get_variables() in self.previous_states.keys():
                    previous_state = self.previous_states[actual_action.get_variables()]
                    self.learn_from_feedback(previous_state, actual_action, actual_utility)

                state.clear_evidence(evidence_var)

        if len(state.get_action_node_ids()) != 0:
            try:
                self.previous_states[state.get_action_node_ids()] = copy(state)
            except Exception as e:
                self.log.warning("cannot copy state: " + str(e))

    @dispatch(DialogueState, Assignment, float)
    def learn_from_feedback(self, state, actual_action, actual_utility):
        """
        Re-estimate the posterior distribution for the domain parameters in the
        dialogue state given the actual system decision and its resulting utility
        (provided by the simulator).

        :param state: the dialogue state
        :param actual_action: the action that was selected
        :param actual_utility: the resulting utility for the action.
        """
        try:
            relevant_params = set()

            for string in state.get_parameter_ids():
                if len(state.get_chance_node(string).get_output_nodes()) != 0:
                    relevant_params.add(string)

            if len(relevant_params) != 0:
                query = UtilQuery(state, relevant_params, actual_action)
                empirical_distrib = self.sampler.get_weighted_samples(query, lambda x: self.reweight_samples(x, actual_utility))

                for param in relevant_params:
                    param_node = self.system.get_state().get_chance_node(param)
                    new_distrib = empirical_distrib.get_marginal(param, param_node.get_input_node_ids())
                    param_node.set_distrib(new_distrib)

            pass
        except Exception as e:
            self.log.warning("could not learn from action feedback: " + str(e))

    @dispatch(Collection, float)
    def reweight_samples(self, samples, actual_utility):
        """
        Reweight Samples
        :param samples: samples
        :param actual_utility: actual utility
        """
        for s in samples:
            weight = 1.0 / (abs(s.get_utility() - actual_utility) + 1)
            s.add_log_weight(log(weight))
