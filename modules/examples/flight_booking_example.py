import logging
from collections import Collection

from multipledispatch import dispatch

from dialogue_state import DialogueState
from modules.module import Module


class FlightBookingExample(Module):
    """
    Example of simple external module used for the flight-booking dialogue domain. The
    module monitors for two particular values for the system action:

    - "FindOffer" checks the (faked) price of the user order and returns
      MakeOffer(price)
    - "Book" simulates the booking of the user order.
    """

    log = logging.getLogger('PyOpenDial')

    def __init__(self, system):
        from dialogue_system import DialogueSystem
        if not isinstance(system, DialogueSystem):
            raise NotImplementedError("UNDEFINED PARAMETERS")

        """
        Creates a new instance of the flight-booking module

        :param system: the dialogue system to which the module should be attached
        """
        self._system = system
        self._paused = True

    def start(self):
        """
        Starts the module.
        """
        self._paused = False

    @dispatch(DialogueState, Collection)
    def trigger(self, state, update_vars):
        """
        Checks whether the updated variables contains the system action and (if yes)
        whether the system action value is "FindOffer" or "Book". If the value is
        "FindOffer", checks the price of the order (faked here to 179 or 299 EUR) and
        adds the new action "MakeOffer(price)" to the dialogue state. If the value is
        "Book", simply write down the order on the system output.

        :param state: the current dialogue state
        :param update_vars: the updated variables in the state
        """
        if 'a_m' in update_vars and state.has_chance_node('a_m'):
            action = str(state.query_prob('a_m').get_best())

            if action == 'FindOffer':
                return_date = str(state.query_prob('ReturnDate').get_best())

                price = 179 if return_date == 'NoReturn' else 299
                new_action = 'MakeOffer(%d)' % price
                self._system.add_content('a_m', new_action)
            elif action == 'Book':
                departure = str(state.query_prob('Departure').get_best())
                destination = str(state.query_prob('Destination').get_best())
                date = str(state.query_prob('Date').get_best())
                return_date = str(state.query_prob('ReturnDate').get_best())
                nb_tickets = str(state.query_prob('NbTickets').get_best())

                info = 'Booked %s tickets from %s to %s on %s' % (nb_tickets, departure, destination, date)
                if return_date == 'NoReturn':
                    info += ' and return on ' + return_date

                return info

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
