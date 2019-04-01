import logging
import sys
from collections import Collection

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from multipledispatch import dispatch

from bn.distribs.categorical_table import CategoricalTable
from bn.values.none_val import NoneVal
from dialogue_state import DialogueState
from gui.gui import GUI
from modules.module import Module
from utils.string_utils import StringUtils


class GUIFrame(Module):
    """
    Main GUI frame for the OpenDial toolkit, encompassing various tabs and menus to
    control the application
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    def __init__(self, system):
        from dialogue_system import DialogueSystem
        if not isinstance(system, DialogueSystem):
            raise NotImplementedError("UNDEFINED PARAMETERS")

        """
        Constructs (but does not yet display) a new GUI frame for OpenDial.

        :param system: system the dialogue system for the GUI
        """
        self.frame = False
        self.tabbed_pane = None
        self.state_monitor_tab = None
        self.chat_tab = None
        self.editor_tab = None
        self.menu = None
        self._paused = True

        self._system = system
        self.gui = None
        self.is_speech_enabled = False
        self._audio_module = None
        self._settings = system.get_settings()

    @dispatch()
    def start(self):
        """
        Displays the GUI frame.
        """
        if self._system.get_settings().show_gui and not self.is_running():
            self.frame = True
            app = QtWidgets.QApplication(sys.argv)
            self.gui = GUI(self._system)
            if self.is_speech_enabled:
                self.gui.enable_speech(True)

            # if not self._system._domain._xml_file is None:
            #     self.gui.open_domain(self._system._domain)
            #     self._system.start_system()
            sys.exit(app.exec_())

    @dispatch(bool)
    def pause(self, pause):
        """
        Pauses the GUI.
        """
        self._paused = pause

    @dispatch(DialogueState, Collection)
    def trigger(self, state, update_vars):
        for variable in [self._system.get_settings().user_input, self._system.get_settings().system_output]:
            if not self._paused and variable in update_vars and state.has_chance_node(variable):
                if variable in [self._settings.system_output, self._settings.user_input]:
                    table = self._system.get_content(variable).to_discrete()
                    text = self._get_text_rendering(table)
                    self.gui.chatlog.append(text)
                    self.gui.chatlog.setAlignment(Qt.AlignLeft)

    @dispatch(CategoricalTable)
    def _get_text_rendering(self, table):
        """
        Generates the text representation for the categorical table.

        :param table: the table
        :return: the text rendering of the table
        """
        text_table = ''
        base_variable = table.get_variable().replace("'", '')

        if base_variable == self._system.get_settings().user_input:
            text_table += '\n[user]\t'
        elif base_variable == self._system.get_settings().system_output:
            text_table += '[system]\t'
        else:
            text_table += '[' + base_variable + ']\t'

        for value in table.get_values():
            if not isinstance(value, NoneVal):
                content = str(value)
                if table.get_prob(value) < 0.98:
                    content += ' (' + StringUtils.get_short_form(table.get_prob(value)) + ')'

                text_table += content + '\n\t\t'

        if base_variable == self._system.get_settings().user_input:
            text_table += '\n'

        text_table = text_table[0:-3]
        return text_table

    @dispatch()
    def refresh(self):
        pass

    @dispatch()
    def do_refresh(self):
        pass

    @dispatch(DialogueState, str)
    def record_state(self, state, name):
        pass

    @dispatch(str)
    def add_comment(self, comment):
        pass

    @dispatch(bool)
    def enable_speech(self, to_enable):
        self.is_speech_enabled = to_enable

        if self.gui is None:
            return

        self.gui.enable_speech(to_enable)
        # if self.chat_tab is not None:
        #     self.chat_tab.enable_speech(to_enable)
        #
        # if self.menu is not None:
        #     self.menu.enable_speech(to_enable)

    @dispatch(bool)
    def set_saved_flag(self, is_saved):
        pass

    @dispatch(int)
    def set_action_tab(self, i):
        pass

    @dispatch()
    def request_save(self):
        pass

    @dispatch()
    def close_window(self):
        pass

    @dispatch()
    def get_system(self):
        pass

    @dispatch()
    def get_chat_tab(self):
        pass

    @dispatch()
    def get_state_viewer_tab(self):
        pass

    @dispatch()
    def get_editor_tab(self):
        pass

    @dispatch()
    def get_frame(self):
        pass

    @dispatch()
    def is_running(self):
        return self.frame

    @dispatch()
    def is_speech_enabled(self):
        pass

    @dispatch()
    def is_domain_saved(self):
        pass

    @dispatch()
    def get_menu(self):
        pass

    @dispatch()
    def new_domain(self):
        pass

    # TODO: 'File' type
    @dispatch(object)
    def new_domain(self, file_to_save):
        pass

    @dispatch()
    def open_domain(self):
        pass

    @dispatch()
    def save_domain(self):
        pass

    # TODO: 'File' type
    @dispatch(object)
    def save_domain(self, file_to_write):
        pass

    @dispatch()
    def save_domain_as(self):
        pass

    @dispatch()
    def reset_interaction(self):
        pass

    @dispatch(bool)
    def import_interaction(self, is_wizard_of_oz):
        pass

    @dispatch()
    def save_interaction(self):
        pass

    @dispatch(str)
    def import_content(self, tag):
        pass

    @dispatch(str)
    def export_content(self, tag):
        pass


class ClickListener:
    pass
