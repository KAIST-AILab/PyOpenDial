from multipledispatch import dispatch
from bn.values.value import Value
from utils.string_utils import StringUtils
import numpy as np


class ArrayVal(Value):
    """
    Representation of an array of doubles.
    """
    def __init__(self, values):
        if isinstance(values, np.ndarray):
            """
            Creates a new array of values

            :param values: the array
            """
            self._value = np.array(values)
        else:
            raise NotImplementedError()

    def __hash__(self):
        """
        Returns the hashcode for the array.

        :return: the hash code
        """
        return hash(tuple(self._value))

    def __eq__(self, other):
        """
        Returns true if the given argument is same as a value array.

        :param other: the object to compare
        :return: true if the given argument is same as a value array
        """
        if not isinstance(other, ArrayVal):
            return False

        return np.allclose(self._value, other.get_array())

    def __str__(self):
        """
        Returns a string representation of the array.

        :return: the string representation of the array
        """
        return '[' + ','.join([StringUtils.get_short_form(d) for d in self._value]) + ']'

    def __len__(self):
        """
        Returns a length of the array.

        :return: the length of the array
        """
        return len(self._value)

    def __lt__(self, other):
        """
        Compares to another value.

        :param other: the object to compare
        """
        # TODO: check refactor > Is this comparison proper?
        if not isinstance(other, ArrayVal):
            return hash(self) < hash(other)

        if len(self._value) != len(other.val):
            return len(self._value) < len(other.get_array())

        for idx, val in enumerate(self._value):
            other_val = other.val[idx]

            from settings import Settings
            if abs(val - other_val) > Settings.eps:
                return val < other_val

        return False

    def __copy__(self):
        """
        Copies the array.

        :return: the copy
        """
        return ArrayVal(self._value)

    def __contains__(self, item):
        """
        Returns true if the item is in the array.

        :param item: the value
        :return: true if the item is in the array.
        """
        return True if item.get_double() in self._value else False

    @dispatch()
    def get_sub_values(self):
        """
        Returns the list of double values.

        :return: the list of double values
        """
        from bn.values.double_val import DoubleVal
        return [DoubleVal(d) for d in self._value]

    @dispatch(Value)
    def concatenate(self, value):
        """
        If value is an ArrayVal, returns the combined array value. Else, returns none.

        :param value: the value to concatenate
        :return: the concatenated result
        """
        if isinstance(value, ArrayVal):
            return ArrayVal(self._value.extend(value.get_array()))

        from bn.values.none_val import NoneVal
        if isinstance(value, NoneVal):
            return self

        set_values = set()
        for v in self._value:
            from bn.values.value_factory import ValueFactory
            set_values.add(ValueFactory.create(v))

        from bn.values.set_val import SetVal
        result = SetVal(set_values)
        return result.concatenate(value)

    @dispatch()
    def get_array(self):
        """
        Returns the array.

        :return: the array
        """
        return self._value
