import logging
import xml.etree.ElementTree as ET
from collections import Collection

from multipledispatch import dispatch

from dialogue_state import DialogueState
from modules.module import Module
from utils.xml_utils import XMLUtils


class DialogueRecorder(Module):
    """
    Module used to systematically record all user inputs and outputs during the
    interaction. The recordings are stored in a XML element which can be written to a
    file at any time.

    The module can also be used to record Wizard-of-Oz interactions.
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    def __init__(self, system):
        from dialogue_system import DialogueSystem
        if not isinstance(system, DialogueSystem):
            raise NotImplementedError("UNDEFINED PARAMETERS")

        """
        Creates a new dialogue recorder for the dialogue system

        :param system: the dialogue system
        """
        self._settings = system.get_settings()
        self._root_node = None
        self._doc = None

    @dispatch()
    def start(self):
        """
        Starts the recorder.
        """
        try:
            self._doc = XMLUtils.new_xml_document("interaction")
            self._root_node = XMLUtils.get_main_node(self._doc)
        except Exception as e:
            self.log.warning("could not create dialogue recorder")
            raise ValueError()

    @dispatch(bool)
    def pause(self, shouldBePaused):
        """
        Does nothing.
        """
        pass

    @dispatch(DialogueState, Collection)  # Collection: collection of strings
    def trigger(self, state, updated_vars):
        """
        Triggers the recorder with a particular dialogue state and a set of recently
        updated variables. If one of the updated variables is the user input or system
        output, the recorder stores a new turn. Else, the module does nothing.
        """
        if self._root_node.tag != "interaction":
            self.log.warning("root node is ill-formatted: %s or first value is null" % self._root_node.tag)
            raise ValueError()
        # if the user is still speaking, do not record anything yet
        if state.has_chance_node(self._settings.user_speech):
            return

        try:
            if self._settings.user_input in updated_vars:
                vars_to_record = set()
                vars_to_record.add(self._settings.user_input)
                vars_to_record.update(self._settings.vars_to_monitor)
                el = state.generate_xml(vars_to_record)
                if len(list(el)) > 0:
                    el.tag = "userTurn"
                    self._root_node.append(el)
            if self._settings.system_output in updated_vars:
                vars_to_record = set()
                vars_to_record.add(self._settings.system_output)
                vars_to_record.update(self._settings.vars_to_monitor)
                if state.has_chance_node("a_m"):
                    vars_to_record.add("a_m")
                el = state.generate_xml(vars_to_record)
                if len(list(el)) > 0:
                    el.tag = "systemTurn"
                    self._root_node.append(el)

        except Exception as e:
            self.log.warning("cannot record dialogue turn %s" % e)
            raise ValueError()

    @dispatch(str)
    def add_comment(self, comment):
        """
        Adds a comment in the XML recordings.
        :param comment: the comment to add
        """
        try:
            if self._root_node.tag == "interaction":
                com = ET.Comment(comment)
                self._root_node.append(com)
            else:
                self.log.warning("could not add comment")
        except Exception as e:
            self.log.warning("could not record preamble or comment: %s" % e)

    @dispatch(str)
    def write_to_file(self, record_file):
        """
        Write the recorded dialogue to a file

        :param record_file: the pathname for the file
        """
        self.log.debug("recording interaction in file " + record_file)
        try:
            XMLUtils.write_xml_document(self._doc, record_file)
        except Exception as e:
            self.log.warning("could not create file " + record_file)


    @dispatch()
    def get_record(self):
        """
        Serialises the XML recordings and returns the output.
        :return: the serialised XML content.
        """
        return XMLUtils.serialize(self._root_node)

    @dispatch()
    def is_running(self):
        """
        Returns true if the module is running, and false otherwise.
        """
        return self._doc is not None
