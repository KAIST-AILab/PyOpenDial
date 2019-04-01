import logging
from enum import Enum, auto

from multipledispatch import dispatch

from modules.module import Module


class MessageType(Enum):
    """
    types of messages that can be sent through the connector
    """
    INIT = auto()
    XML = auto()
    STREAM = auto()
    MISC = auto()
    CLOSE = auto()


class RemoteConnector(Module):
    """
    Module used to connect OpenDial to other remote clients (for instance, in order to
    conduct Wizard-of-Oz experiments).
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    # ===================================
    # CONSTRUCTION
    # ===================================

    @dispatch(object)  # object: DialogueSystem
    def __init__(self, system):
        """
        A server socket is created, using an arbitrary open port (NB: the port can be
        read in the "About" page in the GUI).

        :param system: the local dialogue system opened
        """
        self.system = system  # the local dialogue system
        self.paused = True  # whether the connector is paused or not
        self.skip_next_trigger = False  # whether to skip the next trigger (to avoid infinite loops)
