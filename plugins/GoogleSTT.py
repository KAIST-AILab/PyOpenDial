import io
import os
import threading
from collections import Collection
from time import sleep

from google.cloud import speech
from multipledispatch import dispatch

from datastructs.speech_data import SpeechData
from dialogue_state import DialogueState
from gui.gui_frame import GUIFrame
from modules.audio_module import AudioModule
from modules.module import Module
from utils.audio_utils import AudioUtils


class GoogleSTT(Module):
    def __init__(self, system):
        from dialogue_system import DialogueSystem
        if not isinstance(system, DialogueSystem):
            raise NotImplementedError("UNDEFINED PARAMETERS")

        self._system = system
        self._stt_client = speech.SpeechClient()
        self._stt_config = speech.types.RecognitionConfig(
            encoding=speech.enums.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code='en-US'
        )

        self._paused = True
        self._lock = threading.RLock()

        self._system.enable_speech(True)

    def start(self):
        self._paused = False
        gui = self._system.get_module(GUIFrame)
        if gui is None:
            raise RuntimeError("Google STT requires access to GUI.")

    @dispatch(DialogueState, Collection)
    def trigger(self, state, updated_vars):
        user_speech_var = self._system.get_settings().user_speech

        if user_speech_var in updated_vars and state.has_chance_node(user_speech_var) and not self._paused:
            speech_val = self._system.get_content(user_speech_var).get_best()
            if isinstance(speech_val, SpeechData):
                def thread_func(speech):
                    self.recognize(speech)

                threading.Thread(target=thread_func, args=(speech_val,)).start()

    @dispatch(bool)
    def pause(self, to_pause):
        self._paused = to_pause

    def is_running(self):
        return not self._paused

    def recognize(self, speech_data):
        with self._lock:
            while not speech_data.is_file_write_done():
                sleep(AudioModule.NUM_FRAMES / AudioModule.SAMPLE_RATE)

            content = AudioUtils.read_tmp_recording()
            audio = speech.types.RecognitionAudio(content=content)

            # Detects speech in the audio file
            response = self._stt_client.recognize(self._stt_config, audio)

            lines = dict()
            for result in response.results:
                for alternative in result.alternatives:
                    line = alternative.transcript
                    confidence = alternative.confidence
                    lines[line] = confidence

            # TODO: check normalize confidences?

            self._system.add_user_input(lines)
