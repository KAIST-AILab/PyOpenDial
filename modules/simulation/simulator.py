import logging
import threading
from collections import Collection
from copy import copy
from time import sleep

from multipledispatch import dispatch

from bn.values.value import Value
from bn.values.value_factory import ValueFactory
from datastructs.assignment import Assignment
from dialogue_state import DialogueState
from domains.domain import Domain
from modules.module import Module
from modules.simulation.reward_learner import RewardLearner
from readers.xml_domain_reader import XMLDomainReader
from utils.string_utils import StringUtils


class Simulator(Module):
    """
    Simulator for the user/environment. The simulator generated new environment
    observations and user utterances based on a dialogue domain.
    """

    log = logging.getLogger('PyOpenDial')

    def __init__(self, arg1=None, arg2=None):
        from dialogue_system import DialogueSystem
        if not isinstance(arg1, DialogueSystem):
            raise NotImplementedError("UNDEFINED PARAMETERS")

        system = arg1
        if isinstance(arg2, str):
            domain = arg2
            """
            Creates a new user/environment simulator.

            :param system: the main dialogue system to which the simulator should connect
            :param domain: the dialogue domain for the simulator simulator could
                           not be created
            """
            domain = XMLDomainReader.extract_domain(domain)
        elif isinstance(arg2, Domain):
            domain = arg2
        else:
            raise NotImplementedError("UNDEFINED PARAMETERS")

        """
        Creates a new user/environment simulator.

        :param system: the main dialogue system to which the simulator should connect
        :param domain: the dialogue domain for the simulator not be created
        """
        self.system = system
        self.domain = domain
        self.simulator_state = copy(domain.get_initial_state())
        self.simulator_state.set_parameters(domain.get_parameters())
        self.system.change_settings(domain.get_settings())

        self._lock = threading.RLock()

    def start(self):
        """
        Adds an empty action to the dialogue system to start the interaction.
        """
        empty_action = Assignment(self.system.get_settings().system_output, ValueFactory.none())

        if self.system.is_paused():
            self.system.get_state().add_to_state(empty_action)
        else:
            self.system.add_content(empty_action)
        self.system.attach_module(RewardLearner)

    def is_running(self):
        """
        Returns true if the system is not paused, and false otherwise
        """
        return False

    @dispatch(bool)
    def pause(self, to_pause):
        """
        Do nothing.
        """
        pass

    @dispatch(DialogueState, Collection)
    def trigger(self, system_state, updated_vars):
        """
        Triggers the simulator by updating the simulator state and generating new
        observations and user inputs.

        :param system_state: the dialogue state of the main dialogue system
        :param updated_vars: the updated variables in the dialogue system
        """
        if self.system.get_settings().system_output in updated_vars:
            # TODO: Thread
            # new Thread(() -> performTurn())).start()
            def thread_func():
                self.perform_turn()
            threading.Thread(target=thread_func).start()

    @dispatch()
    def perform_turn(self):
        system_state = self.system.get_state()
        output_var = self.system.get_settings().system_output

        try:
            system_action = ValueFactory.none()
            if system_state.has_chance_node(output_var):
                system_action = system_state.query_prob(output_var).get_best()
            self.log.debug("Simulator input: %s" % system_action)

            turn_performed = self.perform_turn(system_action)
            repeat = 0
            while not turn_performed and repeat < 5 and self.system.get_modules().contains(self):
                turn_performed = self.perform_turn(system_action)
                repeat += 1

        except Exception as e:
            self.log.debug("cannot update simulator: " + str(e))

    @dispatch(Value)
    def perform_turn(self, system_action):
        """
        Performs the dialogue turn in the simulator.

        :param system_action: the last system action.
        """

        with self._lock:
            turn_performed = False
            self.simulator_state.set_parameters(self.domain.get_parameters())
            system_assign = Assignment(self.system.get_settings().system_output, system_action)
            self.simulator_state.add_to_state(system_assign)

            while len(self.simulator_state.get_new_variables()) > 0:
                to_process = self.simulator_state.get_new_variables()
                self.simulator_state.reduce()

                for model in self.domain.get_models():
                    if model.is_triggered(self.simulator_state, to_process):
                        change = model.trigger(self.simulator_state)
                        if change and model.is_blocking():
                            break

                if len(self.simulator_state.get_utility_node_ids()) > 0:
                    reward = self.simulator_state.query_util()
                    comment = 'Reward: ' + StringUtils.get_short_form(reward)
                    self.system.display_comment(comment)
                    self.system.get_state().add_evidence(Assignment('R(' + system_assign.add_primes() + ')', reward))
                    self.simulator_state.remove_nodes(self.simulator_state.get_utility_node_ids())

                if self.add_new_observations():
                    turn_performed = True

                self.simulator_state.add_evidence(self.simulator_state.get_sample())
            return turn_performed

    def add_new_observations(self):
        """
        Generates new simulated observations and adds them to the dialogue state. The
        method returns true when a new user input has been generated, and false
        otherwise.

        :return: whether a user input has been generated.
        """
        new_obs_vars = []

        for var in self.simulator_state.get_chance_node_ids():
            if "^o'" in var:
                new_obs_vars.append(var)

        if len(new_obs_vars) != 0:
            new_obs = self.simulator_state.query_prob(new_obs_vars)

            for new_obs_var in new_obs_vars:
                new_obs.modify_variable_id(new_obs_var, new_obs_var.replace("^o'", ""))

            while self.system.is_paused():
                try:
                    sleep(0.1)
                except Exception as e:
                    pass

            if len(new_obs.get_values()) != 0:
                if self.system.get_settings().user_input in new_obs.get_variables():
                    self.log.debug("Simulator output: " + str(new_obs) + "\n --------------")
                    self.system.add_content(new_obs)
                    return True

                else:
                    self.log.debug("Contextual variables: " + str(new_obs))
                    self.system.add_content(new_obs)

        return False
