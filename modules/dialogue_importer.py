import logging

from threading import Thread
from time import sleep

from multipledispatch import dispatch

from dialogue_state import DialogueState
from modules.dialogue_recorder import DialogueRecorder
from modules.forward_planner import ForwardPlanner


class DialogueImporter(Thread):
    """
    Functionality to import a previously recorded dialogue in the dialogue system. The
    import essentially "replays" the previous interaction, including all state update
    operations.
    """
    # logger
    log = logging.getLogger('PyOpenDial')

    def __init__(self, system, turns):
        """
        Creates a new dialogue importer attached to a particular dialogue system, and
        with an ordered list of turns (encoded by their dialogue state).

        :param system: the dialogue system
        :param turns: the sequence of turns
        """
        self.system = system
        self.turns = turns
        self.wizard_of_mode = False

    @dispatch(bool)
    def set_wizard_of_oz_mode(self, is_wizard_of_oz):
        """
        Sets whether the import should consider the system actions as "expert"
        Wizard-of-Oz actions to imitate.
        
        :param is_wizard_of_oz: whether the system actions are wizard-of-Oz examples
        """
        self.wizard_of_mode = is_wizard_of_oz

    @dispatch()
    def run(self):
        if self.wizard_of_mode:
            # TODO: WizardLearner
            # self.system.attach_module(WizardLearner)
            # for turn in self.turns:
            #     self.add_turn(turn)
            pass
        else:
            self.system.detach_module(ForwardPlanner)
            for turn in self.turns:
                self.add_turn(turn)
                self.system.get_state().remove_nodes(self.system.get_state().get_action_node_ids())
                self.system.get_state().remove_nodes(self.system.get_state().get_utility_node_ids())
            self.system.attach_module(ForwardPlanner)

    @dispatch(DialogueState)
    def add_turn(self, turn):
        try:
            while self.system.is_pauesd() or not self.system.get_module(DialogueRecorder).is_running():
                try:
                    # TODO: Thread
                    sleep(100)
                except:
                    pass
            self.system.add_content(turn.copy())
        except Exception as e:
            self.log.warning("could not add content: %s" % e)
