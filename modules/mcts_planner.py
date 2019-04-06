import logging
from copy import copy
from math import *
from random import *

import numpy as np

from bn.distribs.distribution_builder import MultivariateTableBuilder
from datastructs.assignment import Assignment
from modules.module import Module
from settings import Settings

# logger
logger = logging.getLogger('PyOpenDial')


class StateNode():
    """
    Constructs a state node for the mstc algorithm.

    """

    def __init__(self, state):
        self.state = state
        self.value = 0
        self.visit_count = 0
        self.child_nodes = []
        self.initialized = False
        self.rewards = None

        action_nodes = state.get_action_node_ids()
        if len(action_nodes) != 0:
            rewards = state.query_util(action_nodes)
            self.rewards = rewards
            for action in rewards.get_rows():
                self.child_nodes.append(ActionNode(action, initial_value=rewards.get_util(action)))

    def add_child(self, action_node):
        self.child_nodes.append(action_node)

    def has_child(self, action_node):
        if len(self.child_nodes) == 0:
            return False
        for child in self.child_nodes:
            if child.action == action_node.action:
                return True
        return False

    def uct_select_child(self, c):
        """
        Select the next state based on UCB rule
        :return:
        """
        best_action_node = self.child_nodes[0]

        for action_node in self.child_nodes:
            if action_node.visit_count == 0:
                best_action_node = action_node
                break

            if best_action_node.value + c * sqrt(log(self.visit_count) / best_action_node.visit_count)\
                    < action_node.value + c * sqrt(log(self.visit_count) / action_node.visit_count):
                best_action_node = action_node

        return best_action_node

    def get_best_child(self):
        best_child = self.child_nodes[0]
        logger.debug(self.child_nodes)

        for child in self.child_nodes:
            if child.value > best_child.value:
                best_child = child

        return best_child


class ActionNode():
    """
    Constructs a action node for the mstc algorithm.

    """
    def __init__(self, action, initial_value=0):
        self.action = action
        self.value = initial_value
        self.visit_count = 0
        self.child_nodes = []

    def add_child(self, state_node):
        """
        Add the child Node
        :return:
        """
        self.child_nodes.append(state_node)

    def has_child(self, state_node):
        if len(self.child_nodes) == 0:
            return False
        for child in self.child_nodes:
            if child.state == state_node.state:
                return True
        return False

    def find_child(self, state_node):
        for child in self.child_nodes:
            if child.state == state_node.state:
                return child

    def __str__(self):
        return "(ActionNode=%s/Q=%.3f,N=%d)" % (str(self.action), self.value, self.visit_count)

    __repr__ = __str__


class MCTS():
    def __init__(self, init_state, system):
        self.root = init_state
        self.system = system
        self.simulation_count = system.get_settings().mcts_simulation_count
        self.max_depth = system.get_settings().horizon - 1
        self.gamma = system.get_settings().discount_factor
        self.c = system.get_settings().mcts_exploration_constant

    def get_reward(self, state_node, action_node):
        action = action_node.action

        if len(state_node.child_nodes) != 0:
            reward = state_node.rewards.get_util(action)
        else:
            reward = 0

        return reward

    def next_state_node(self, state_node, action_node):
        state = copy(state_node.state)
        action = copy(action_node.action)

        state.add_to_state(action.remove_primes())
        self.update_state(state)

        observations = self.get_observations(state)
        observations_list = list(observations.get_values())
        obs_probs = []

        for obs in observations.get_values():
            obs_probs.append(observations.get_prob(obs))

        obs = np.random.choice(observations_list, 1, p=obs_probs)[0]

        state.add_to_state(obs)
        self.update_state(state)

        next_state_node = StateNode(state)

        return next_state_node

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

    def update_state(self, state):
        while len(state.get_new_variables()) > 0:
            to_process = state.get_new_variables()
            state.reduce()
            for model in self.system.get_domain().get_models():
                if model.is_triggered(state, to_process):
                    change = model.trigger(state)
                    if change and model.is_blocking():
                        break

    def rollout(self, state_node, depth):
        if depth > self.max_depth or len(state_node.child_nodes) == 0:
            return 0

        action_node = state_node.child_nodes[randint(0, len(state_node.child_nodes)-1)]
        next_state_node = self.next_state_node(state_node, action_node)
        reward = self.get_reward(state_node, action_node)

        terminal = state_node.state.has_chance_node('Terminal') and \
                   state_node.state.get_chance_node('Terminal').sample().get_boolean()

        if terminal:
            return reward

        # simulate
        R = reward + self.gamma * self.rollout(next_state_node, depth+1)
        return R

    def simulate(self, state_node, depth):
        if depth > self.max_depth or len(state_node.child_nodes) == 0:
            return 0
        action_node = state_node.uct_select_child(self.c)

        if not state_node.has_child(action_node):
            state_node.add_child(action_node)
        action_node.visit_count += 1

        next_state_node = self.next_state_node(state_node, action_node)
        reward = self.get_reward(state_node, action_node)

        if action_node.has_child(next_state_node):
            next_state_node = action_node.find_child(next_state_node)
        else:
            action_node.add_child(next_state_node)

        state_node.visit_count += 1

        terminal = state_node.state.has_chance_node('Terminal') and \
                   state_node.state.get_chance_node('Terminal').sample().get_boolean()

        if terminal:
            return reward

        if state_node.initialized:
            # simulate
            R = reward + self.gamma * self.simulate(next_state_node, depth+1)

            # update V, Q values...
            state_node.value = (state_node.value * (state_node.visit_count - 1) + R) / state_node.visit_count
            action_node.value = (action_node.value * (action_node.visit_count - 1) + R) / action_node.visit_count
            return R

        else:
            state_node.initialized = True
            # simulate
            R = reward + self.gamma *  self.rollout(next_state_node, depth+1)
            return R

    def search(self):
        for i in range(self.simulation_count):
            self.simulate(self.root, 0)

        best_action = self.root.get_best_child().action

        return best_action


class MCTSPlanner(Module):

    def __init__(self, system):
        """
        Constructs a mcts planner for the dialogue system.

        :param system:
        """
        self.system = system
        self.current_process = None
        self.paused = False

    def pause(self, should_be_paused):
        """
        Pause the mcts planner
        """
        self.paused = should_be_paused
        if self.current_process != None and not self.current_process.is_terminated:
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


class PlannerProcess():
    """
    Planner process, which is for the mcts algorithm
    """

    def __init__(self, init_state, system, paused):
        """
        Creates the planning process. Timeout is set to twice the maximum sampling
        ime. Then, runs the planner for the number of simulations, or the
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

        # step 1: find the best action with mcts
        state_copy = copy(init_state)
        state_copy.add_to_state(Assignment("__planning", '__planning'))
        state_copy.get_chance_node("__planning'").set_id("__planning")

        init_state_node = StateNode(state_copy)

        planner = MCTS(init_state_node, self.system)
        best_action = planner.search()

        # step 2: remove the action and utility nodes
        init_state.remove_nodes(init_state.get_utility_node_ids())
        action_vars = init_state.get_action_node_ids()
        init_state.remove_nodes(action_vars)

        # step 3: add the selection action to the dialogue state
        init_state.add_to_state(best_action.remove_primes())
        self.is_terminated = True
