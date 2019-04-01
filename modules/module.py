import abc
from collections import Collection

from multipledispatch import dispatch

from dialogue_state import DialogueState


class Module:
    """
    Representation of a system module. A module is connected to the dialogue system
    and can read and write to its dialogue state. It can also be paused/resumed.

    Two distinct families of modules can be distinguished:
     - Asynchronous modules run independently of the dialogue system (once initiated
       by the method start().
     - Synchronous modules are triggered upon an update of the dialogue state via the
       method trigger(state, updatedVars).

    Of course, nothing prevents in practice a module to operate both in synchronous
    and asynchronous mode.

    In order to make the module easy to load into the system (via e.g. the
    "&lt;modules&gt;" parameters in system settings or the command line), it is a good
    idea to ensure that implement each module with a constructor with a single
    argument: the DialogueSystem object to which it should be connected. Additional
    arguments can in this case be specified through parameters in the system settings.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def start(self):
        """
        Starts the module.
        """
        raise NotImplementedError()

    @dispatch(DialogueState, Collection)
    @abc.abstractmethod
    def trigger(self, state, updated_vars):
        """
        Triggers the module after a state update

        :param state: the dialogue state
        :param updated_vars: the set of updated variables
        """
        raise NotImplementedError()

    @dispatch(bool)
    @abc.abstractmethod
    def pause(self, to_pause):
        """
        Pauses the current module

        :param to_pause: whether to pause or resume the module
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def is_running(self):
        """
        Returns true if the module is running (i.e. started and not paused), and false
        otherwise

        :return: whether the module is running or not
        """
        raise NotImplementedError()
