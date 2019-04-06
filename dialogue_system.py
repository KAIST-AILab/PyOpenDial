import logging
import threading
from collections import Collection
from copy import copy

from multipledispatch import dispatch

from bn.b_network import BNetwork
from bn.distribs.distribution_builder import CategoricalTableBuilder
from bn.distribs.independent_distribution import IndependentDistribution
from bn.distribs.multivariate_distribution import MultivariateDistribution
from bn.distribs.prob_distribution import ProbDistribution
from bn.values.value import Value
from datastructs.assignment import Assignment
from datastructs.speech_data import SpeechData
from dialogue_state import DialogueState
from domains.domain import Domain
from gui.gui_frame import GUIFrame
from modules.audio_module import AudioModule
from modules.dialogue_importer import DialogueImporter
from modules.dialogue_recorder import DialogueRecorder
from modules.forward_planner import ForwardPlanner
from modules.mcts_planner import MCTSPlanner
from modules.module import Module
from readers.xml_dialogue_reader import XMLDialogueReader
from readers.xml_domain_reader import XMLDomainReader
from settings import Settings
from utils.py_utils import get_class_name_from_type, get_class_name


class DialogueSystem:
    """
    Dialogue system based on probabilistic rules. A dialogue system comprises:
    - the current dialogue state
    - the dialogue domain with a list of rule-structured models
    - the list of system modules
    - the system settings.

    After initialising the dialogue system, the system should be started with the
    method startSystem(). The system can be paused or resumed at any time.
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    def __init__(self, arg1=None):

        if arg1 is None:
            """
            Creates a new dialogue system with an empty dialogue system
            """
            self._settings = Settings()  # the system setting
            self._cur_state = DialogueState()  # the dialogue state

            self._domain = Domain()  # the dialogue domain
            self._paused = True  # whether the system is paused or active
            self._modules = []  # the set of modules attached to the system

            # Inserting standard modules
            system = self
            self._modules.append(GUIFrame(system))
            self._modules.append(DialogueRecorder(self))
            if self._settings.planner == 'forward':
                self.log.info("Forward planner will be used.")
                self._modules.append(ForwardPlanner(self))
            elif self._settings.planner == 'mcts':
                self.log.info("MCTS planner will be used.")
                self._modules.append(MCTSPlanner(self))
            else:
                raise ValueError("Not supported planner: %s" % self._settings.planner)
            self._init_lock()

        elif isinstance(arg1, Domain):
            domain = arg1
            """
            Creates a new dialogue system with the provided dialogue domain

            :param domain: the dialogue domain to employ
            """
            self.__init__()
            self.change_domain(domain)
        elif isinstance(arg1, str):
            domain_file = arg1
            """
            Creates a new dialogue system with the provided dialogue domain

            :param domain_file: the dialogue domain to employ
            """
            self.__init__()
            self.change_domain(XMLDomainReader.extract_domain(domain_file))

    def _init_lock(self):
        # TODO: need refactoring (decorator?)
        self._locks = {
            'detach_module': threading.RLock(),
            'start_system_update': threading.RLock(),
            'pause_update': threading.RLock(),
            'update': threading.RLock()
        }

    @dispatch()
    def start_system(self):
        """
        Starts the dialogue system and its modules.
        """
        self._paused = False
        for module in self._modules:
            try:
                if not module.is_running():
                    module.start()
                else:
                    module.pause(False)

            except Exception as e:
                self.log.warning("could not start module %s: %s" % (type(module), e))
                self._modules.remove(module)

        with self._locks['start_system_update']:
            self._cur_state.set_as_new()
            self.update()

    @dispatch(Domain)
    def change_domain(self, domain):
        """
        Changes the dialogue domain for the dialogue domain

        :param domain: the dialogue domain to employ
        """
        self._domain = domain
        self.change_settings(domain.get_settings())
        self._cur_state = copy(domain.get_initial_state())
        self._cur_state.set_parameters(domain.get_parameters())

        if not self._paused:
            self.start_system()

    @dispatch(Module)
    def attach_module(self, module_instance):
        """
        Attaches the module to the dialogue system.

        :param module_instance: the module to add
        """
        if module_instance in self._modules or self.get_module(module_instance) is not None:
            self.log.info("Module %s is already attached" % type(module_instance))
            return

        if len(self._modules) == 0:
            self._modules.append(module_instance)
        else:
            self._modules.insert(len(self._modules) - 1, module_instance)

        if not self._paused:
            try:
                module_instance.start()
            except Exception as e:
                self.log.warning("could not start module %s" % type(module_instance))
                self._modules.remove(module_instance)

    @dispatch(type)
    def attach_module(self, module_type):
        """
        Attaches the module to the dialogue system.

        :param module_type: the module class to instantiate
        """
        try:
            module_instance = module_type(self)
            self.attach_module(module_instance)
            self.display_comment("Module %s successfully attached" % module_type.__name__)
        except Exception as e:
            self.log.warning("cannot attach %s: %s" % (module_type.__name__, e))
            self.display_comment("cannot attach %s: %s" % (module_type.__name__, e))

    @dispatch(type)
    def detach_module(self, module_type):
        """
        Detaches the module of the dialogue system. If the module is not included in the system, does nothing.
        Only one of model_type or module_instance will be given as an input parameter.

        :param module_type: the class of the module to detach.
        :param module_instance: the module to detach
        """
        with self._locks['detach_module']:
            module_instance = self.get_module(module_type)
            if module_instance is not None:
                module_instance.pause(True)
                self._modules.remove(module_instance)

    @dispatch(bool)
    def pause(self, to_pause):
        """
        Pauses or resumes the dialogue system.

        :param to_pause: whether the system should be paused or resumed.
        """
        self._paused = to_pause

        for module in self._modules:
            module.pause(to_pause)

        if not to_pause and not self._cur_state.get_new_variables().is_empty():
            with self._locks['pause_update']:
                self.update()

    @dispatch(str)
    def display_comment(self, comment):
        """
        Adds a comment on the GUI and the dialogue recorder.
        :param comment: comment the comment to display
        """
        if self.get_module(GUIFrame) is not None and self.get_module(GUIFrame).is_running():
            self.get_module(GUIFrame).add_comment(comment)
        else:
            self.log.info(comment)
        if self.get_module(DialogueRecorder) is not None and self.get_module(DialogueRecorder).is_running():
            self.get_module(DialogueRecorder).add_comment(comment)

    @dispatch(Settings)
    def change_settings(self, settings):
        """
        Changes the settings of the system

        :param settings: the new settings
        """
        self._settings.fill_settings(settings.get_specified_mapping())

        for module_type in settings.modules:
            if self.get_module(module_type) is None:
                self.attach_module(module_type)

    @dispatch(bool)
    def enable_speech(self, to_enable):
        if to_enable:
            if self.get_module(AudioModule) is None:
                self._settings.select_audio_mixers()
                self.attach_module(AudioModule(self))
                if self._settings.show_gui:
                    self.get_module(GUIFrame).enable_speech(True)
                else:
                    raise NotImplementedError()
                    # TODO: VAD not implemented
                    # self.get_module(type(AudioModule)).activate_vad(True)
        else:
            self.detach_module(AudioModule)
            if self.get_module(GUIFrame) is not None:
                self.get_module(GUIFrame).enable_speech(False)

    @dispatch(str)
    def import_dialogues(self, dialogue_file):
        turns = XMLDialogueReader.extract_dialogue(dialogue_file)
        importer = DialogueImporter(self, turns)
        importer.start()
        return importer

    # ===============================
    # STATE UPDATE
    # ===============================

    @dispatch(str)
    def add_user_input(self, user_input):
        """
        Adds the user input (assuming a perfect confidence score) to the dialogue
        state and subsequently updates it.

        :param user_input: the user input as a string
        :return: the variables that were updated in the process not be updated
        """
        # perfect confidence score
        a = Assignment(self._settings.user_input, user_input)
        return self.add_content(a)

    @dispatch(dict)
    def add_user_input(self, user_input):
        """
        Adds the user input (as a N-best list, where each hypothesis is associated
        with a probability) to the dialogue state and subsequently updates it.

        :param user_input: the user input as an N-best list
        :return: the variables that were updated in the process not be updated
        """
        # user_input: N-best list, where each hypothesis is associated with a probability
        var = self._settings.user_input if not self._settings.inverted_role else self._settings.system_output

        builder = CategoricalTableBuilder(var)

        for input in user_input.keys():
            builder.add_row(input, user_input.get(input))

        return self.add_content(builder.build())

    @dispatch(SpeechData)
    def add_user_input(self, input_speech):
        assignment = Assignment(self._settings.user_speech, input_speech)
        assignment.add_pair(self._settings.floor, 'user')
        return self.add_content(assignment)

    @dispatch(str, (str, bool, Value, float))
    def add_content(self, variable, value):
        """
        Adds the content (expressed as a pair of variable=value) to the current
        dialogue state, and subsequently updates the dialogue state.

        :param variable: the variable label
        :param value: the variable value
        :return: the variables that were updated in the process not be updated.
        """
        if not self._paused:
            self._cur_state.add_to_state(Assignment(variable, value))
            return self.update()

        else:
            self.log.info("System is paused, ignoring %s = %s" % (variable, value))
            return set()

    @dispatch((Assignment, IndependentDistribution, ProbDistribution, MultivariateDistribution, BNetwork, DialogueState))
    def add_content(self, distrib):
        """
        Merges the dialogue state included as argument into the current one, and
        updates the dialogue state.

        :param distrib: the content to add
        :return: the set of variables that have been updated
        """
        if not self._paused:
            self._cur_state.add_to_state(distrib)
            return self.update()
        else:
            self.log.info("System is paused, ignoring content %s" % distrib)
            return set()

    @dispatch(IndependentDistribution, bool)
    def add_incremental_content(self, content, follow_previous):
        """
        Adds the incremental content (expressed as a distribution over variables) to
        the current dialogue state, and subsequently updates it. If followPrevious is
        set to true, the content is concatenated with the current distribution for the
        variable.

        :param content: the content to add / concatenate
        :param follow_previous: whether the results should be concatenated to the previous values,
                                or reset the content (e.g. when starting a new utterance)
        :return: the set of variables that have been updated update failed
        """
        if not self._paused:
            self._cur_state.add_to_state_incremental(content.to_discrete(), follow_previous)
            return self.update()

        else:
            self.log.info("System is paused, ignoring content " % content)
            return set()

    @dispatch(dict, bool)
    def add_incremental_user_input(self, user_input, follow_previous):
        """
        Adds the incremental user input (expressed as an N-best list) to the current
        dialogue state, and subsequently updates it. If followPrevious is set to true,
        the content is concatenated with the current distribution for the variable.
        This allows (for instance) to perform incremental updates of user utterances.

        :param user_input: the user input to add / concatenate
        :param follow_previous: whether the results should be concatenated to the previous values,
                                or reset the content (e.g. when starting a new
                                utterance)
        :return: the set of variables that have been updated update failed
        """
        builder = CategoricalTableBuilder(self._settings.user_input)

        for input in user_input.key_set():
            builder.add_row(input, user_input.get(input))

        return self.add_incremental_content(builder.build(), follow_previous)

    @dispatch()
    def remove_content(self, variable_id):
        """
        Removes the variable from the dialogue state

        :param variable_id: the variable identifier
        """
        if not self._paused:
            self._cur_state.remove_from_state(variable_id)
            self.update()

        else:
            self.log.info("System is paused, ignoring removal of %s" % variable_id)

    @dispatch()
    def update(self):
        """
        Performs an update loop on the current dialogue state, by triggering all the
        models and modules attached to the system until all possible updates have been
        performed. The dialogue state is pruned at the end of the operation.

        :return: the set of variables that have been updated during the process.
        """

        with self._locks['update']:
            updated_vars = dict()

            while len(self._cur_state.get_new_variables()) > 0:
                to_process = self._cur_state.get_new_variables()

                self._cur_state.reduce()

                for model in self._domain.get_models():
                    if not model.planning_only and model.is_triggered(self._cur_state, to_process):
                        change = model.trigger(self._cur_state)
                        if change and model.is_blocking():
                            break

                for i in range(len(self._modules)):
                    self._modules[i].trigger(self._cur_state, to_process)

                for v in to_process:
                    if v not in updated_vars or updated_vars[v] is None:
                        count = 1
                    else:
                        count = updated_vars[v] + 1
                    updated_vars[v] = count

                    if count > 100:  # TODO: count > 10 ?
                        self.display_comment("Warning: Recursive update of variable %s" % v)
                        return set(updated_vars.keys())

        return set(updated_vars.keys())

    @dispatch()
    def refresh_domain(self):
        """
        Refreshes the dialogue domain by rereading its source file (in case it has been changed by the user).
        """
        if self._domain.is_empty():
            return

        src_file = self._domain.get_source_file().get_path()

        try:
            self._domain = XMLDomainReader.extract_domain(src_file)
            self.change_settings(self._domain.get_settings())
            self.display_comment("Dialogue domain successfully updated")
        except Exception as e:
            self.log.critical("Cannot refresh domain %s" % e)
            self.display_comment("Syntax error: %s" % e)
            self._domain = Domain()
            self._domain.set_source_file(src_file)

    # ===============================
    # GETTERS
    # ===============================

    @dispatch()
    def get_state(self):
        """
        Returns the current dialogue state for the dialogue system.

        :return: the dialogue state
        """
        return self._cur_state

    @dispatch()
    def get_floor(self):
        """
        Returns who holds the current conversational floor (user, system, or free)

        :return: a string stating who currently owns the floor
        """
        if self._cur_state.has_chance_node(self._settings.floor):
            return str(self.get_content(self._settings.floor).get_best())
        else:
            return "free"

    @dispatch((str, Collection))
    def get_content(self, variable):
        """
        Returns the probability distribution associated with the variables in the current dialogue state.

        :param variable: the variable to query, which will be 'str' or 'list'
        :return: the resulting probability distribution for these variables
        """
        if isinstance(variable, list):
            variable = set(variable)

        return self._cur_state.query_prob(variable)

    @dispatch(type)
    def get_module(self, module_type):
        """
        Returns the module attached to the dialogue system and belonging to a
        particular class, if one exists. If no module exists, returns null

        :param module_type: the module class
        :return: the attached module of that class, if one exists.
        """
        for module in self._modules:
            module_name = get_class_name_from_type(module_type)
            if get_class_name(module) == module_name:
                return module
        return None

    @dispatch(Module)
    def get_module(self, module_instance):
        """
        Returns the module attached to the dialogue system and belonging to a
        particular class, if one exists. If no module exists, returns null

        :param module_instance: the module instance
        :return: the attached module of that class, if one exists.
        """
        for module in self._modules:
            module_name = get_class_name(module_instance)
            if get_class_name(module) == module_name:
                return module
        return None

    @dispatch(type, Module)
    def get_module(self, module_type, module_instance):
        """
        Returns the module attached to the dialogue system and belonging to a
        particular class, if one exists. If no module exists, returns null

        :param module_type: the module class
        :param module_instance: the module instance
        :return: the attached module of that class, if one exists.
        """
        for module in self._modules:
            module_name = get_class_name(module_instance)
            if get_class_name(module) == module_name:
                return module
        return None

    @dispatch()
    def is_paused(self):
        """
        Returns true is the system is paused, and false otherwise

        :return: true if paused, false otherwise.
        """
        return self._paused

    @dispatch()
    def get_settings(self):
        """
        Returns the settings for the dialogue system.

        :return: the system settings.
        """
        return self._settings

    @dispatch()
    def get_domain(self):
        """
        Returns the domain for the dialogue system.

        :return: the dialogue domain.
        """
        return self._domain

    @dispatch()
    def get_modules(self):
        """
        Returns the collection of modules attached to the system.

        :return: the modules (list).
        """
        return self._modules
