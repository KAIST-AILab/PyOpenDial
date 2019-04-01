import logging
import threading
from collections import Collection
from time import sleep

import numpy as np
import soundcard as sc
from multipledispatch import dispatch

from datastructs.assignment import Assignment
from datastructs.speech_data import SpeechData
from dialogue_state import DialogueState
from modules.module import Module
from utils.audio_utils import AudioUtils


class AudioModule(Module):
    """
    Module used to take care of all audio processing functionalities in OpenDial. The
    module is employed both to record audio data from the microphone and to play audio
    data on the system speakers.

    Two modes are available to record audio data:
    - a manual mode when the user explicitly click on the "Press and hold to record speech"
      to indicate the start and end points of speech data.
    - an automatic mode relying on (energy-based) Voice Activity Recognition to determine when speech is present
      in the audio stream.

    When speech is detected using one of the two above methods, the module creates a
    SpeechData object containing the captured audio stream and updates the dialogue
    state with a new value for the variable denoting the user speech (by default s_u).
    This data is then presumably picked up by a speech recogniser for further
    processing.

    The module is also used for the reverse operation, namely playing audio data
    (generated via e.g. speech synthesis) on the target audio line. When the module
    detects a new value for the variable denoting the system speech (by default s_m),
    it plays the corresponding audio on the target line.

    The module can gracefully handle user interruptions (when the user starts speaking
    when the system is still talking).
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    # Threshold for the difference between the current and background audio volume
    # level above which the audio is considered as speech
    VOLUME_THRESHOLD = 250

    # Minimum duration for a sound to be considered as possible speech (in
    # milliseconds)
    MIN_DURATION = 300

    # file used to save the speech input (leave empty to avoid recording)
    SAVE_SPEECH = ""

    SAMPLE_RATE = 16000

    NUM_FRAMES = SAMPLE_RATE / 5

    def __init__(self, system):
        from dialogue_system import DialogueSystem
        if isinstance(system, DialogueSystem):
            """
            Creates a new audio recorder connected to the dialogue system.
            """
            # the dialogue system
            self.system = system
            # The microphone for capturing audio
            self.microphone = None
            # The speaker for speech
            self.speaker = None
            # The recorded speech (null if the input audio is not currently speech)
            self.input_speech = None
            # The output speech currently playing
            self.output_speech = []
            # whether the module is paused or not
            self.is_paused = True
            # # whether the speech is to be automatically detected or not
            # self.voice_activity_detection = False
            # # current audio level
            # self.current_volume = 0.0
            # # background audio level
            # self.background_volume = 0.0
            # speech panel (used to e.g. show the current volume)
            # self.speech_panel = None
        else:
            raise NotImplementedError()

    @dispatch()
    def start(self):
        """
        Starts the audio recording
        """
        self.is_paused = False
        self.microphone = sc.default_microphone()
        self.speaker = sc.default_speaker()

    # @dispatch(SpeechInputPanel)
    # def attach_panel(self, speech_panel):
    #     """
    #     Attaches the speech panel to the module
    #
    #     :param speech_panel: the speech input panel (containing the volume panel)
    #     """
    #     self.speech_panel = speech_panel

    # @dispatch(bool)
    # def activate_vad(self, activate_vad):
    #     """
    #     Activates or deactivates voice activity detection (VAD). If VAD is
    #     deactivated, the GUI button "press and hold to record speech" is used to mark
    #     the start and end points of speech data.
    #
    #     :param activate_vad: true if VAD should be activated, false otherwise
    #     """
    #     self.voice_activity_detection = activate_vad

    @dispatch()
    def start_recording(self):
        """
        Starts the recording of a new speech segment, and adds its content to the
        dialogue state. If voice activity recognition is used, the new speech segment
        is only inserted after waiting a minimum duration, in order to avoid inserting
        many spurious short noises into the dialogue state. Otherwise, the speech is
        inserted immediately.
        """
        if not self.is_paused:
            # creates a new SpeechData object
            self.input_speech = SpeechData(np.zeros((0, 1)))

            # state update procedure
            def state_update():
                if self.input_speech is not None and not self.input_speech.is_final():
                    self.system.add_user_input(self.input_speech)

            # TODO: run recording thread
            # def thread_func():
            #     sleep(self.MIN_DURATION)
            #     state_update()

            # performs the update
            # if self.voice_activity_detection:
            #     threading.Thread(target=thread_func).start()
            # else:

            def recording_thread():
                data = np.zeros((0, 1))
                while not self.input_speech.is_final():
                    data = np.r_[data, self.microphone.record(samplerate=self.SAMPLE_RATE, numframes=self.NUM_FRAMES, channels=1, blocksize=2)]

                self.input_speech.write(data)
                if len(self.input_speech) > self.MIN_DURATION:
                    AudioUtils.write_tmp_recording(self.input_speech.data)
                    self.input_speech.set_as_file_write_done()

                self.system.add_content(self.system.get_settings().floor, "free")

            state_update()
            threading.Thread(target=recording_thread).start()
        else:
            self.log.info("Audio recorder is currently paused")

    @dispatch()
    def stop_recording(self):
        """
        Stops the recording of the current speech segment.
        """
        if self.input_speech is not None:
            self.input_speech.set_as_complete()

    @dispatch(DialogueState, Collection)
    def trigger(self, state, updated_vars):
        """
        Checks whether the dialogue state contains a updated value for the system
        speech (by default denoted as s_m). If yes, plays the audio on the target
        line.
        """
        system_speech = self.system.get_settings().system_speech
        if system_speech in updated_vars and state.has_chance_node(system_speech):
            speech = state.query_prob(system_speech).get_best()
            if isinstance(speech, SpeechData):
                print('speaker trigger')
                self.system.add_content(Assignment(self.system.get_settings().floor, "system"))
                self.play_speech(speech)

    @dispatch(SpeechData)
    def play_speech(self, speech):
        """
        Plays the speech data onto the default target line.
        :param sound: the sound to play
        """
        if len(self.output_speech) == 0:
            self.output_speech.append(speech)

            def speaker_thread():
                while len(self.output_speech) > 0:
                    speech_data = self.output_speech.pop(0).data
                    self.speaker.play(speech_data, samplerate=self.SAMPLE_RATE)

                self.system.add_content(self.system.get_settings().floor, "free")

            threading.Thread(target=speaker_thread).run()

        else:
            self.output_speech.append(speech)
            # if the system is already playing a sound, concatenate to the existing one
            # self.output_speech = self.output_speech.concatenate(speech)

    @dispatch(bool)
    def pause(self, to_pause):
        """
        Pauses the recorder
        """
        self.is_paused = to_pause

    @dispatch()
    def is_running(self):
        """
        Returns true if the recorder is currently running, and false otherwise
        """
        return not self.is_paused

