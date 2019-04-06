import logging
from collections import Collection

from multipledispatch import dispatch

from dialogue_state import DialogueState
from example_domains.negotiation.negotiation_functions import generate_user_utterance, update_dialogue_history, \
    generate_user_selection
from modules.module import Module


class NegotiationUserModule(Module):

    log = logging.getLogger('PyOpenDial')

    def __init__(self, system):
        from dialogue_system import DialogueSystem
        if not isinstance(system, DialogueSystem):
            raise NotImplementedError("UNDEFINED PARAMETERS")
        print('NEGO USER MODULE')

        self._system = system
        self._paused = True

        self.prev_u_m = None
        self.user_selection = None

    def start(self):
        """
        Starts the module.
        """
        print('NEGO USER MODULE STARTED')
        self._paused = False

    @dispatch(DialogueState, Collection)  # Collection: collection of strings
    def trigger(self, state, update_vars):
        negotiation_state = state.query_prob('negotiation_state').get_best().get_value()
        u_m = str(state.query_prob('u_m').get_best())
        current_step = str(state.query_prob('current_step').get_best())
        dialogue_history = str(state.query_prob('dialogue_history').get_best())

        if ('current_step' in update_vars and current_step in ['Negotiation', 'Selection']) or ('u_m' in update_vars and u_m != self.prev_u_m):
            if current_step == 'Negotiation' and 'selection' not in u_m:
                dialogue_history = update_dialogue_history(dialogue_history, 'system', u_m)
                u_u = generate_user_utterance(negotiation_state, dialogue_history, u_m)
                print()
                print('****************************')
                print('dialogue_history: ', dialogue_history.split("\n"))
                print('u_u:', u_u)
                print('****************************')
                print()
            elif current_step == 'Selection' and self.user_selection is None:
                u_u = generate_user_selection(negotiation_state, dialogue_history)
                self.user_selection = u_u
                print()
                print('****************************')
                print('dialogue_history: ', dialogue_history.split("\n"))
                print('u_u:', u_u)
                print('****************************')
                print()

            self.prev_u_m = u_m

    @dispatch(bool)
    def pause(self, to_pause):
        """
        Pauses the module.

        :param to_pause: whether to pause the module or not
        """
        self._paused = to_pause

    def is_running(self):
        """
        Returns whether the module is currently running or not.

        :return: whether the module is running or not.
        """
        return not self._paused
