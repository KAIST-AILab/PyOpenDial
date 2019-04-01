import logging
from collections import Collection
from copy import copy

from multipledispatch import dispatch

from bn.distribs.distribution_builder import MultivariateTableBuilder
from bn.distribs.utility_table import UtilityTable
from datastructs.assignment import Assignment
from dialogue_state import DialogueState
from modules.module import Module
from settings import Settings


def process_utility_table(utility_table, target_action_ids):
    target_utility_table = UtilityTable(utility_table.get_additional_info(),
                                        utility_table.get_apply_changes_func(),
                                        utility_table.get_rollback_changes_func())
    other_assignments = []
    for assignment in utility_table.get_rows():
        action_ids = assignment.get_variables()
        utility = utility_table.get_util(assignment)
        if action_ids == target_action_ids:
            target_utility_table.set_util(assignment, utility)
        else:
            # print('=================')
            # print('other assignments', assignment)
            # print('action_ids: ', action_ids)
            # print('target_ids: ', target_action_ids)
            # print('===================')

            other_assignments.append(assignment)
    return target_utility_table, other_assignments


def apply_action_to_state(state, action, nontarget_assignments):
    state.add_to_state(action.remove_primes())
    for assignment in nontarget_assignments:
        state.add_to_state(assignment.remove_primes())


class ForwardPlanner(Module):
    """
    Online forward planner for OpenDial. The planner constructs a lookahead tree (with
    a depth corresponding to the planning horizon) that explores possible actions and
    their expected consequences on the future dialogue state. The final utility values
    for each action is then estimated, and the action with highest utility is
    selected.

    The planner is an anytime process. It can be interrupted at any time and yield a
    result. The quality of the utility estimates is of course improving over time.

    The planning algorithm is described in pages 121-123 of Pierre Lison's PhD thesis
    [http://folk.uio.no/plison/pdfs/thesis/thesis-plison2013.pdf]

    @author Pierre Lison (plison@ifi.uio.no)
    """

    log = logging.getLogger('PyOpenDial')

    # Maximum number of actions to consider at each planning step
    nb_best_actions = 100
    # Maximum number of alternative observations to consider at each planning step
    nb_best_observations = 3
    # Minimum probability for the generated observations
    min_observation_prob = 0.1

    # TODO: ScheduledExecutorService
    # service = Executors.newScheduledThreadPool(2);

    def __init__(self, system):
        from dialogue_system import DialogueSystem
        if not isinstance(system, object): # object: DialogueSystem
            raise NotImplementedError("UNDEFINED PARAMETERS")

        """
        Constructs a forward planner for the dialogue system.

        :param system: the dialogue system associated with the planner.
        """
        self.system = system
        self.current_process = None
        self.paused = False

    @dispatch(bool)
    def pause(self, should_be_paused):
        """
        Pause the forward planner
        """
        self.paused = should_be_paused
        if self.current_process is not None and not self.current_process.is_terminated:
            self.current_process.is_terminated = True

    def start(self):
        """
        Does nothing
        """
        pass

    def is_running(self):
        """
        :return: true if the planner is not paused.
        """
        return not self.paused

    @dispatch(DialogueState, Collection)
    def trigger(self, state, updated_vars):
        """
        Triggers the planning process.
        """
        # disallows action selection while the user is still talking
        if self.system.get_floor() == "user":
            state.remove_nodes(state.get_action_node_ids())
            state.remove_nodes(state.get_utility_node_ids())

        if not self.paused and not len(state.get_action_node_ids()) == 0:
            self.current_process = PlannerProcess(state, self.system, self.paused)


class PlannerProcess:
    """
    Planner process, which can be terminated before the end of the horizon

    @author Pierre Lison (plison@ifi.uio.no)
    """

    def __init__(self, init_state, system, paused):
        from dialogue_system import DialogueSystem
        if not isinstance(init_state, DialogueState) or not isinstance(system, DialogueSystem) or not isinstance(paused, bool):
            raise NotImplementedError("UNDEFINED PARAMETERS")

        """
        Creates the planning process. Timeout is set to twice the maximum sampling
        ime. Then, runs the planner until the horizon has been reached, or the
        planner has run out of time. Adds the best action to the dialogue state.

        :param init_state: initial dialogue state
        """
        self.system = system
        self.paused = paused

        self.is_terminated = False

        settings = system.get_settings()
        # setting the timeout for the planning
        timeout = Settings.max_sampling_time * 2

        # if the speech stream is not finished, only allow fast, reactive responses
        if init_state.has_chance_node(settings.user_speech):
            timeout = timeout / 5.0

        # TODO: ScheduledExecutorService
        # service.schedule(() -> isTerminated = true, timeout, TimeUnit.MILLISECONDS);

        # handle custom utility (rename action node from var to simvar)
        if init_state.get_custom_utility_function() and init_state.get_custom_utility_function().get_action_node_id() + "'" in init_state.get_action_node_ids():
            action_node_id = init_state.get_custom_utility_function().get_action_node_id()
            simulation_action_node_id = init_state.get_custom_utility_function().get_simulation_action_node_id()
            init_state.remove_node(action_node_id + "'")
            if not init_state.has_action_node(simulation_action_node_id):
                import bn.nodes.action_node
                simulation_action_node = bn.nodes.action_node.ActionNode(simulation_action_node_id + "'")
                init_state.add_node(simulation_action_node)
            else:
                raise ValueError("Simulation action node already exists")

        # step 1: extract the Q-values
        eval_actions = self.get_q_values(init_state, settings.horizon)
        target_action_ids = init_state.get_action_node_ids()
        eval_actions_target, nontarget_assignments = process_utility_table(eval_actions, target_action_ids)

        # step 2: find the action with highest utility
        best_action = eval_actions_target.get_best()[0]
        if eval_actions_target.get_util(best_action) < 0.001:
            best_action = Assignment.create_default(best_action.get_variables())

        # handle custom utility (rename action node from simvar to var)
        if init_state.get_custom_utility_function() and init_state.get_custom_utility_function().get_simulation_action_node_id() + "'" in best_action.get_variables():
            best_action = best_action.rename_var(
                init_state.get_custom_utility_function().get_simulation_action_node_id() + "'",
                init_state.get_custom_utility_function().get_action_node_id() + "'"
            )

        # apply changes
        if eval_actions.get_apply_changes_func() is not None:
            eval_actions.apply_changes(init_state, best_action)

        # step 3: remove the action and utility nodes
        init_state.remove_nodes(init_state.get_utility_node_ids())
        action_vars = init_state.get_action_node_ids()
        init_state.remove_nodes(action_vars)

        # step 4: add the selection action to the dialogue state
        apply_action_to_state(init_state, best_action, nontarget_assignments)
        self.is_terminated = True

    @dispatch(DialogueState, int)
    def get_q_values(self, state, horizon):
        """
        Returns the Q-values for the dialogue state, assuming a particular horizon.

        :param state: the dialogue state
        :param horizon: the planning horizon
        :return: the estimated utility table for the Q-values
        """
        action_nodes = state.get_action_node_ids()

        if len(action_nodes) == 0:
            return UtilityTable()

        raw_rewards = state.query_util(action_nodes)
        # print(rewards)

        if horizon == 1:
            return raw_rewards

        q_values = UtilityTable(raw_rewards.get_additional_info(),
                                raw_rewards.get_apply_changes_func(),
                                raw_rewards.get_rollback_changes_func())

        discount = self.system.get_settings().discount_factor

        target_action_ids = action_nodes
        _, nontarget_assignments = process_utility_table(raw_rewards, target_action_ids)
        for action in raw_rewards.get_rows():
            reward = raw_rewards.get_util(action)
            if target_action_ids == action.get_variables():
                q_values.set_util(action, reward)

                if horizon > 1 and not self.is_terminated and not self.paused and self.has_transition(action):
                    state_copy = copy(state)
                    # PART 1. apply changes
                    if raw_rewards.get_apply_changes_func() is not None:
                        values = raw_rewards.apply_changes(state, action)

                    apply_action_to_state(state_copy, action, nontarget_assignments)
                    self.update_state(state_copy)

                    if not action.is_default():
                        # PART 2. simulate
                        expected = discount * self.get_expected_value(state_copy, horizon - 1)
                        # PART 3. rollback
                        if raw_rewards.get_rollback_changes_func() is not None:
                            raw_rewards.rollback_changes(state, action, values)
                        # ----------------
                        q_values.set_util(action, q_values.get_util(action) + expected)
            else:
                q_values.set_util(action, reward)

        return q_values

    @dispatch(DialogueState)
    def update_state(self, state):
        """
        Adds a particular content to the dialogue state

        :param state: the dialogue state
        """
        while len(state.get_new_variables()) > 0:
            to_process = state.get_new_variables()
            state.reduce()

            for model in self.system.get_domain().get_models():
                if model.is_triggered(state, to_process):
                    change = model.trigger(state)
                    if change and model.is_blocking():
                        break

    @dispatch(Assignment)
    def has_transition(self, action):
        """
        Returns true if the dialogue domain specifies a transition model for the
        particular action assignment.

        :param action: the assignment of action values.
        :return: true if a transition is defined, false otherwise.
        """
        for m in self.system.get_domain().get_models():
            if m.is_triggered(action.remove_primes().get_variables()):
                return True
        return False

    @dispatch(DialogueState, int)
    def get_expected_value(self, state, horizon):
        """
        Estimates the expected value (V) of the dialogue state in the current planning horizon.

        :param state: the dialogue state
        :param horizon: the planning horizon
        :return: the expected value.
        """
        observations = self.get_observations(state)
        nbest_obs = observations.get_n_best(ForwardPlanner.nb_best_observations)
        expected_value = 0.0

        for obs in nbest_obs.get_values():
            obs_prob = nbest_obs.get_prob(obs)
            if obs_prob > ForwardPlanner.min_observation_prob:
                state_copy = copy(state)
                state_copy.add_to_state(obs)
                self.update_state(state_copy)

                q_values = self.get_q_values(state_copy, horizon)
                if len(q_values.get_rows()) > 0:
                    best_action = q_values.get_best()[0]
                    after_obs = q_values.get_util(best_action)
                    expected_value += obs_prob * after_obs

        return expected_value

    @dispatch(DialogueState)
    def get_observations(self, state):
        """
        Returns the possible observations that are expected to be perceived from the dialogue state

        :param state: the dialogue state from which to extract observations
        :return: the inferred observations
        """
        prediction_nodes = set()

        for node_id in state.get_chance_node_ids():
            if "^p" in node_id:
                prediction_nodes.add(node_id)

        # intermediary observations
        for node_id in prediction_nodes:
            if state.get_chance_node(node_id).has_descendant(prediction_nodes):
                prediction_nodes.remove(node_id)

        builder = MultivariateTableBuilder()

        if len(prediction_nodes) != 0:
            observations = state.query_prob(prediction_nodes)

            for a in observations.get_values():
                new_a = Assignment()
                for var in a.get_variables():
                    new_a.add_pair(var.replace("^p", ""), a.get_value(var))

                builder.add_row(new_a, observations.get_prob(a))

        return builder.build()
