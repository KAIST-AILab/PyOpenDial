import io
import logging
import os

import soundfile as sf
import numpy as np

from multipledispatch import dispatch

dispatch_namespace = dict()


class AudioFormat:

    @dispatch(float, int, int, bool, bool)
    def __init__(self, sample_rate, sample_size_in_bits, channels, signed, big_endian):
        self.sample_rate = sample_rate
        self.sample_size_in_bits = sample_size_in_bits
        self.channels = channels
        self.signed = signed
        self.big_endian = big_endian


class AudioUtils:
    log = logging.getLogger('PyOpenDial')

    # Audio format for higher-quality speech recognition (frame rate: 16 kHz)
    IN_HIGH = AudioFormat(16000., 16, 1, True, False)

    # Audio format for lower-quality speech recognition (frame rate: 8 kHz)
    IN_LOW = AudioFormat(8000., 16, 1, True, False)

    # Audio format for the speech synthesis
    OUT = AudioFormat(16000., 16, 1, True, False)

    # Maximum number of samples to consider for the calculation of the root mean-square
    MAX_SIZE_RMS = 100

    TMP_DIR = 'tmp'
    TMP_RECORDING_FILE_NAME = 'speech_recording.wav'
    TMP_GOOGLE_SPEECH_FILE_NAME = 'speech_google.wav'

    @staticmethod
    @dispatch(np.ndarray)
    def write_tmp_recording(data):
        if not os.path.exists(AudioUtils.TMP_DIR):
            os.makedirs(AudioUtils.TMP_DIR)

        file_path = os.path.join(AudioUtils.TMP_DIR, AudioUtils.TMP_RECORDING_FILE_NAME)
        sf.write(file_path, data / np.max(data), 16000)

    @staticmethod
    @dispatch()
    def read_tmp_recording():
        if not os.path.exists(AudioUtils.TMP_DIR):
            os.makedirs(AudioUtils.TMP_DIR)

        file_path = os.path.join(AudioUtils.TMP_DIR, AudioUtils.TMP_RECORDING_FILE_NAME)
        with io.open(file_path, 'rb') as audio_file:
            return audio_file.read()

    @staticmethod
    @dispatch(object)
    def convert_google_response_to_numpy_array(response):
        if not os.path.exists(AudioUtils.TMP_DIR):
            os.makedirs(AudioUtils.TMP_DIR)

        file_path = os.path.join(AudioUtils.TMP_DIR, AudioUtils.TMP_GOOGLE_SPEECH_FILE_NAME)

        with open(file_path, 'wb') as out:
            out.write(response.audio_content)

        speech, _ = sf.read(file_path)
        os.remove(file_path)
        return np.reshape(speech, (speech.shape[0], 1))
