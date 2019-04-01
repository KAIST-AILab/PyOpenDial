import logging
# from pynput.mouse import Listener


class SpeechInputPanel(Listener):
    """
    Panel employed to capture audio input through a press and hold button, accompanied
    by a sound level meter. The captured sound is then sent to the dialogue system for
    further processing by the speech recognition engine.
    """
    # logger
    log = logging.getLogger('PyOpenDial')

    def __init__(self, recorder):
        self.recorder = recorder
        recorder.attach_panel(self)


        pass
