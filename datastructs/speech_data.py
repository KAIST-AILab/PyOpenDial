import logging
import threading
import numpy as np

from time import sleep

from multipledispatch import dispatch

from bn.values.value import Value


class SpeechData(Value):
    """
    Representation of a stream of speech data (input or output). The stream can both
    be read (using the usual methods), but can also be modified by appending new data
    to the end of the stream.

    The stream is allowed to change until it is marked as "final" (i.e. when the audio
    capture has finished recording).
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    def __init__(self, arg1=None):
        if arg1 is None:
            """
            Creates a new, empty stream of data with a given audio format

            :param format: the audio format to employ
            """
            self.data = None
            # self.format = None  # TODO: implement this
            self._is_complete = False
            self._is_file_write_done = False
            self._init_lock()
        elif isinstance(arg1, np.ndarray):
            data = arg1
            """
            Creates a stream of speech data based on a pre-existing byte array.

            :param data: the byte array
            """
            self.data = data
            self._is_complete = False
            self._is_file_write_done = False
            self._init_lock()
        else:
            raise NotImplementedError()

    def _init_lock(self):
        # TODO: need refactoring (decorator?)
        self._locks = {
            'set_as_complete': threading.RLock(),
            'file_write_done': threading.RLock()
        }

    @dispatch()
    def set_as_complete(self):
        """
        Marks the speech data as final (it won't be changed anymore)
        """
        with self._locks['set_as_complete']:
            self._is_complete = True

    @dispatch()
    def set_as_file_write_done(self):
        """
        Marks the speech data as final (it won't be changed anymore)
        """
        with self._locks['file_write_done']:
            self._is_file_write_done = True

    @dispatch()
    def is_final(self):
        return self._is_complete

    @dispatch()
    def is_file_write_done(self):
        return self._is_file_write_done

    @dispatch(np.ndarray)
    def write(self, data):
        self.data = data

    # @dispatch(bytes)
    # def write(self, buffer):
    #     """
    #     Expands the current speech data by appending a new buffer of audio data
    #
    #     :param buffer: the new audio data to insert
    #     """
    #     if self._is_final:
    #         self.log.warning("attempting to write to a final SpeechData object")
    #         return
    #     new_data = bytearray(len(self.data) + len(buffer))
    #     new_data[:len(self.data)] = self.data
    #     new_data[len(self.data):] = buffer
    #     self.data = bytes(new_data)

    # @dispatch()
    # def read(self):
    #     """
    #     Reads one byte of the stream
    #
    #     :return: the read byte
    #     """
    #     if self.current_pos < len(self.data):
    #         result = self.data[self.current_pos]
    #         self.current_pos += 1
    #         return result
    #     else:
    #         if not self._is_final:
    #             try:
    #                 sleep(0.1)
    #             except Exception as e:
    #                 pass
    #             return self.read()
    #         return -1
    #
    # @dispatch(bytes, int, int)
    # def read(self, buffer, offset, length):
    #     """
    #     Reads a buffer of data from the stream
    #
    #     :param buffer: the buffer to fill
    #     :param offset: the offset at which to start filling the buffer
    #     :param length: the maximum number of bytes to read
    #     """
    #     if self.current_pos >= len(self.data):
    #         if self._is_final:
    #             return -1
    #         else:
    #             try:
    #                 sleep(0.02)
    #             except:
    #                 pass
    #     i = 0
    #     buffer = bytearray(buffer)
    #     while True:
    #         if not (i < length and (self.current_pos + i) < len(self.data)):
    #             break
    #         buffer[offset + i] = self.data[self.current_pos + i]
    #         i += 1
    #     self.current_pos += i
    #     return i
    #
    # @dispatch()
    # def rewind(self):
    #     """
    #     Resets the current position in the stream to 0.
    #     """
    #     self.current_pos = 0

    # ===================================
    # GETTERS
    # ===================================

    def __len__(self):
        """
        Returns the duration of the audio data (in milliseconds)
        """
        from modules.audio_module import AudioModule
        return int(len(self.data) / AudioModule.SAMPLE_RATE * 1000)

    @dispatch()
    def is_final(self):
        """
        Returns true if the speech data is final, false otherwise
        :return: true if the data is final, false otherwise
        """
        return self._is_complete

    # @dispatch()
    # def to_byte_array(self):
    #     """
    #     Returns the raw array of bytes
    #     :return: the byte array
    #     """
    #     return self.data

    @dispatch()
    def get_format(self):
        raise NotImplementedError()

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __lt__(self, other):
        return hash(self) < hash(other)

    def __hash__(self):
        """
        Retursn a fixed number (32).

        :return: 32.
        """
        return 32

    def __str__(self):
        """
        Returns a string representation of the data

        :return: the string
        """
        return "Speech data (size: %.1f kb.)" % (len(self.data) / 1000)

    def __copy__(self):
        """
        Returns itself
        """
        return self

    def __contains__(self, subvalue):
        """
        Returns false
        """
        return False

    @dispatch()
    def get_sub_values(self):
        """
        Returns an empty list
        :return:
        """
        return []

    @dispatch(np.ndarray)
    def concatenate(self, speech_data):
        self.data = np.r_[self.data, speech_data]

    # @dispatch(Value)
    # def concatenate(self, speech_data):
    #     """
    #     Returns the concatenation of the two audio data. If the values are not final,
    #     waits for them to be final.
    #     """
    #     if isinstance(value, SpeechData):
    #         while not self.is_final() or not value.is_final():
    #             try:
    #                 sleep(0.05)
    #             except:
    #                 pass
    #
    #         new_data = SpeechData()
    #         new_data.current_pos = self.current_pos
    #         new_data.write(self.data)
    #         new_data.write(value.data)
    #         new_data._is_final = True
    #         return new_data
    #     else:
    #         raise ValueError("Cannot concatenate SpeechData and %s" % type(value).__name__)
