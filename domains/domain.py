from bn.b_network import BNetwork
from dialogue_state import DialogueState
from domains.model import Model
from settings import Settings
from pathlib import Path

import logging
from multipledispatch import dispatch


class Domain:
    """
    Representation of a dialogue domain, composed of (1) an initial dialogue state and
    (2) a list of probability and utility models employed to update the dialogue state
    upon relevant changes.
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    def __init__(self):
        """
        Creates a new domain with an empty dialogue state and list of models.
        """
        self._settings = Settings()
        self._models = []  # list of models
        self._initial_state = DialogueState()  # initial dialog state
        self._parameters = BNetwork()
        self._imported_files = []
        self._xml_file = None  # path to the source XML file (and its imports)

    @dispatch(Path)
    def set_source_file(self, xml_file):
        """
        Associate the given source XML files to the domain.

        :param xml_file: the file to associate
        """
        if xml_file.is_file():
            self._xml_file = xml_file

    @dispatch()
    def is_empty(self):
        """
        Returns true if the domain is empty.

        :return: true if empty, false otherwise
        """
        return not self._xml_file

    @dispatch(Path)
    def add_imported_files(self, xml_file):
        """
        Adds the given XML files to the list of imported source files.

        :param xml_file: the file to add
        """
        if xml_file.is_file():
            self._imported_files.append(xml_file)

    @dispatch()
    def get_source_file(self):
        """
        Returns the source file containing the domain specification.

        :return: the source file
        """
        return self._xml_file

    @dispatch()
    def get_imported_files(self):
        """
        Returns the (possibly empty) list of imported files.

        :return: the imported files
        """
        return self._imported_files

    @dispatch(DialogueState)
    def set_initial_state(self, initial_state):
        """
        Sets the initial dialogue state.

        :param initial_state: the initial state
        """
        self._initial_state = initial_state

    @dispatch(Model)
    def add_model(self, model):
        """
        Adds a model to the domain

        :param model: the model to add
        """
        self._models.append(model)

    @dispatch()
    def get_initial_state(self):
        """
        Returns the initial dialogue state.

        :return: the initial state
        """
        return self._initial_state

    @dispatch()
    def get_models(self):
        """
        Returns the models for the domain.

        :return: the models
        """
        return self._models

    @dispatch(Settings)
    def set_settings(self, settings):
        """
        Replaces the domain-specific settings.

        :param settings: the settings for the domain
        """
        self._settings = settings

    @dispatch()
    def get_settings(self):
        """
        Returns the domain-specific settings.

        :return: the settings for the domain
        """
        return self._settings

    def __str__(self):
        """
        Returns the domain name.

        :return: domain name
        """
        return self._xml_file.name

    @dispatch(BNetwork)
    def set_parameters(self, parameters):
        """
        Sets the prior distribution for the domain parameters.

        :param parameters: the parameters
        """
        self._parameters = parameters

    @dispatch()
    def get_parameters(self):
        """
        Returns the prior distribution for the domain parameters.

        :return: the prior distribution for the parameters
        """
        return self._parameters

    def __eq__(self, other):
        """
        Returns true if o is a domain with the same source file, and false otherwise.

        :param other: the instance to be compared
        :return: true if the instances are equals, false otherwise
        """
        if not isinstance(other, Domain):
            return False

        src = other.get_source_file()
        if src == self._xml_file:
            return True
        return src and self._xml_file and src == self._xml_file
