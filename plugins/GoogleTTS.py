import threading

import soundcard as sc

from collections import Collection

from google.cloud import texttospeech
from multipledispatch import dispatch

from bn.values.string_val import StringVal
from datastructs.assignment import Assignment
from datastructs.speech_data import SpeechData
from dialogue_state import DialogueState
from modules.module import Module
from utils.audio_utils import AudioUtils


class GoogleTTS(Module):
    def __init__(self, system):
        from dialogue_system import DialogueSystem
        if not isinstance(system, DialogueSystem):
            raise NotImplementedError("UNDEFINED PARAMETERS")

        self._system = system
        self._tts_client = texttospeech.TextToSpeechClient()
        self._speaker = sc.default_speaker()

        self._system.enable_speech(True)
        self._is_paused = True
        self._lock = threading.RLock()

        self._voice = texttospeech.types.VoiceSelectionParams(
            language_code='en-US',
            ssml_gender=texttospeech.enums.SsmlVoiceGender.NEUTRAL
        )
        self._audio_config = texttospeech.types.AudioConfig(
            audio_encoding=texttospeech.enums.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000
        )

    def start(self):
        self._is_paused = False

    @dispatch(DialogueState, Collection)
    def trigger(self, state, updated_vars):
        system_output = self._system.get_settings().system_output
        if system_output in updated_vars and state.has_chance_node(system_output) and not self._is_paused:
            utterance_val = state.query_prob(system_output).to_discrete().get_best()

            if isinstance(utterance_val, StringVal):
                def thread_func(utterance):
                    self.synthesise(str(utterance))

                threading.Thread(target=thread_func, args=(utterance_val,)).start()

    def pause(self, to_pause):
        self._is_paused = to_pause

    def is_running(self):
        return not self._is_paused

    def synthesise(self, utterance):
        print(utterance)
        with self._lock:
            synthesis_input = texttospeech.types.SynthesisInput(text=utterance)

            response = self._tts_client.synthesize_speech(synthesis_input, self._voice, self._audio_config)
            speech = AudioUtils.convert_google_response_to_numpy_array(response)
            # self._speaker.play(speech, samplerate=16000)
            output = SpeechData(speech)

            def thread_func():
                self._system.add_content(Assignment(self._system.get_settings().system_speech, output))

            threading.Thread(target=thread_func).start()
            output.set_as_complete()
