import logging
from collections import Collection

from multipledispatch import dispatch

from bn.distribs.categorical_table import CategoricalTable
from bn.values.none_val import NoneVal
from dialogue_state import DialogueState
from modules.module import Module
from utils.string_utils import StringUtils


class TextOnlyInterface(Module):
    """
    Text-only interface to OpenDial, to use when no X11 display is available.
    """

    log = logging.getLogger('PyOpenDial')

    def __init__(self, system):
        from dialogue_system import DialogueSystem
        if not isinstance(system, DialogueSystem):
            raise NotImplementedError("UNDEFINED PARAMETERS")

        """
        Creates a new text-only interface.

        :param system: the dialogue system
        """
        self._system = system
        self._paused = True

    def start(self):
        """
        Starts the interface.
        """
        self._paused = False

    # def _thread_func(self):
    #     while True:
    #         print('Type new input: ')
    #         input_str = input()
    #         table = StringUtils.get_table_from_input(input_str)
    #         if not self._paused and not len(table) == 0:
    #             self._system.add_user_input(table)

    @dispatch(DialogueState, Collection)
    def trigger(self, state, update_vars):
        """
        Updates the interface with the new content (if relevant).

        :param state: dialogue state
        :param update_vars: update variables
        """
        for variable in [self._system.get_settings().user_input, self._system.get_settings().system_output]:
            if not self._paused and variable in update_vars and state.has_chance_node(variable):
                print(self._get_text_rendering(self._system.get_content(variable).to_discrete()))

    @dispatch(bool)
    def pause(self, to_pause):
        """
        Pauses the interface
        """
        self._paused = to_pause

    def is_running(self):
        """
        Returns true if the interface is running, and false otherwise.
        """
        return not self._paused

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
