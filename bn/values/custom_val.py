from bn.values.value import Value
from copy import copy
import logging


class CustomVal(Value):
    """
    Representation of a custom value.
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    def __init__(self, value):
        self._value = value
        self._template = None

    def __hash__(self):
        return hash(self._value)

    def __eq__(self, other):
        if not isinstance(other, CustomVal):
            return False

        return self._value == other._value

    def __lt__(self, other):
        if not isinstance(other, CustomVal):
            return False

        return self._value < other._value

    def __str__(self):
        return str(self._value)

    def __len__(self):
        return 1

    def __copy__(self):
        return self
        # raise NotImplementedError()
        # return CustomVal(copy(self._value))

    def __contains__(self, item):
        return self == item

    def get_sub_values(self):
        raise NotImplementedError()

    def concatenate(self, value):
        raise NotImplementedError()

    def get_value(self):
        return self._value
