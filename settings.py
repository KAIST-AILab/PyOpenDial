import codecs
from xml.etree.ElementTree import Element

import yaml

from utils.py_utils import get_class, get_class_name_from_type
from collections import Callable
import logging
from multipledispatch import dispatch
import soundcard as sc

class Settings:
    """
    System-wide settings for OpenDial.
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    discretization_buckets = 50
    eps = 1e-6
    nr_samples = 3000
    max_sampling_time = 250  # in milliseconds

    _functions = dict()

    def __init__(self, arg1=None):
        if arg1 is None:
            """
            Creates new settings with the default values
            """
            self.show_gui = False
            self.floor = ''
            self.user_speech = ''
            self.system_speech = ''
            self.user_input = ''
            self.system_output = ''
            self.vars_to_monitor = list()
            self.horizon = 0
            self.discount_factor = 0.
            self.inverted_role = False
            self.modules = []
            self.mcts_simulation_count = 1
            self.mcts_exploration_constant = 1.
            self.planner = 'forward'
            self.GOOGLE_APPLICATION_CREDENTIALS = None

            self._input_mixer = None
            self._output_mixer = None

            self._params = dict()
            self._explicit_settings = set()
            self.fill_settings(yaml.load(codecs.open('settings.yml', 'r')))
            self._explicit_settings.clear()
        elif isinstance(arg1, dict):
            mapping = arg1
            """
            Creates a new settings with the values provided as argument. Values that are
            not explicitly specified in the mapping are set to their default values.

            :param mapping: the properties
            """
            self.floor = ''
            self.user_speech = ''
            self.system_speech = ''
            self.user_input = ''
            self.system_output = ''
            self.vars_to_monitor = list()
            self.horizon = 0
            self.discount_factor = 0.
            self.inverted_role = False
            self.modules = []
            self.GOOGLE_APPLICATION_CREDENTIALS = None

            self._input_mixer = None
            self._output_mixer = None

            self._params = dict()
            self._explicit_settings = set()
            self.fill_settings(yaml.load(codecs.open('settings.yml', 'r')))
            self._explicit_settings.clear()

            if mapping is not None:
                self.fill_settings(mapping)
        else:
            raise NotImplementedError()

    def select_audio_mixers(self):
        self._input_mixer = sc.all_speakers()[0] if len(sc.all_speakers()) > 0 else None
        self._output_mixer = sc.all_microphones()[0] if len(sc.all_microphones()) > 0 else None

    @dispatch(dict)
    def fill_settings(self, settings):
        """
        Fills the current settings with the values provided as argument. Existing
        values are overridden.

        :param settings: the properties
        """
        for key, value in settings.items():
            if key.lower() == 'horizon':
                self.horizon = value
            elif key.lower() == 'discount':
                self.discount_factor = value
            elif key.lower() == 'gui':
                self.show_gui = value
            elif key.lower() == 'user':
                self.user_input = value
            elif key.lower() == 'speech_user':
                self.user_speech = value
            elif key.lower() == 'speech_system':
                self.system_speech = value
            elif key.lower() == 'floor':
                self.floor = value
            elif key.lower() == 'system':
                self.system_output = value
            elif key.lower() == 'monitor':
                split = value.split(",")
                for i in range(len(split)):
                    if len(split[i].strip()) > 0:
                        self.vars_to_monitor.append(split[i].strip())
            elif key.lower() == 'samples':
                Settings.nr_samples = value
            elif key.lower() == 'timeout':
                Settings.max_sampling_time = value
            elif key.lower() == 'discretisation':
                Settings.discretization_buckets = value
            elif key.lower() == 'modules' or key.lower() == 'module':
                for module in value.split(','):
                    self.modules.append(get_class(module))
            elif key.lower() == 'mcts_simulation_count':
                self.mcts_simulation_count = value
            elif key.lower() == 'mcts_exploration_constant':
                self.mcts_exploration_constant = float(value)
            elif key.lower() == 'planner':
                if value.lower() in ['forward', 'mcts']:
                    self.planner = value.lower()
                else:
                    raise ValueError("Not supported planner: %s" % value)
            elif key.upper() == 'GOOGLE_APPLICATION_CREDENTIALS':
                self.GOOGLE_APPLICATION_CREDENTIALS = value
            else:
                self._params[key.lower()] = value

            self._explicit_settings.add(key.lower())

    @dispatch()
    def get_full_mapping(self):
        """
        Returns a representation of the settings in terms of a mapping between
        property labels and values
        :return: the corresponding mapping
        """
        mapping = dict()
        mapping.update(self._params)
        mapping["horizon"] = self.horizon
        mapping["discount"] = self.discount_factor
        mapping["speech_user"] = self.user_speech
        mapping["speech_system"] = self.system_speech
        mapping["floor"] = self.floor
        mapping["user"] = self.user_input
        mapping["system"] = self.system_output
        mapping['input_mixer'] = self._input_mixer.id if self._input_mixer else ''
        mapping['output_mixer'] = self._output_mixer.id if self._output_mixer else ''
        # mapping.setProperty("monitor", StringUtils.join(varsToMonitor, ","));
        mapping["monitor"] = ",".join(self.vars_to_monitor)
        mapping["samples"] = Settings.nr_samples
        mapping["timeout"] = Settings.max_sampling_time
        mapping["discretisation"] = Settings.discretization_buckets
        mapping['modules'] = ','.join([get_class_name_from_type(module_type) for module_type in self.modules])
        return mapping

    @dispatch()
    def get_specified_mapping(self):
        full_mapping = self.get_full_mapping()
        mapping = dict()
        for key in full_mapping.keys():
            if key in self._explicit_settings:
                mapping[key] = full_mapping[key]
        return mapping

    @staticmethod
    @dispatch(str, Callable)
    def add_function(name, func):
        Settings._functions[name] = func

    @staticmethod
    @dispatch(str)
    def is_function(str_val):
        for name in Settings._functions.keys():
            name = name.strip()
            if str_val.startswith(name) and str_val[len(name)] == '(' and str_val[-1] == ')':
                return True
        return False

    @staticmethod
    @dispatch(str)
    def get_function(name):
        if name in Settings._functions:
            return Settings._functions[name]
        else:
            raise ValueError()

    def generateXML(self):
        root = Element('settings')

        for key, value in self.get_full_mapping():
            param_element = Element(key)
            param_element.text(str(value))
            root.append(param_element)

        return root
