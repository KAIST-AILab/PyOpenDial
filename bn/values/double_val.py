from bn.values.value import Value
from utils.string_utils import StringUtils
import logging
from multipledispatch import dispatch


class DoubleVal(Value):
    """
    Representation of a double value.
    """
    eps = 1e-10

    # logger
    log = logging.getLogger('PyOpenDial')

    def __init__(self, value):
        if isinstance(value, float):
            """
            Creates the double value (protected, use the ValueFactory instead).

            :param value: the double
            """
            self._value = value
        else:
            raise NotImplementedError()

    def __hash__(self):
        """
        Returns the hashcode for the double.

        :return: the hashcode
        """
        return hash(self._value)

    def __eq__(self, other):
        """
        Returns true if the objects are similar, false otherwise.

        :param other: the object to compare
        :return: true if similar, false otherwise
        """
        if not isinstance(other, DoubleVal):
            return False

        if abs(self._value - other.get_double()) > DoubleVal.eps:
            return False

        return True

    def __lt__(self, other):
        """
        Compares the double value to another value.

        :param other: the object to compare
        :return: usual ordering, or hashcode difference if the value is not a double
        """
        if not isinstance(other, DoubleVal):
            return False

        return self._value < other.get_double()

    def __str__(self):
        """
        Returns a string representation of the double.

        :return: the string representation
        """
        return StringUtils.get_short_form(self._value)

    def __len__(self):
        """
        Returns 1.

        :return: 1
        """
        return 1

    def __copy__(self):
        """
        Returns a copy of the double value.

        :return: the copy
        """
        return DoubleVal(self._value)

    def __contains__(self, item):
        """
        Returns true if the item is same value

        :param item: the value
        :return: true if the item is same value
        """
        return self == item

    @dispatch()
    def get_sub_values(self):
        """
        Returns an empty list.

        :return: an empty list
        """
        return []

    @dispatch(Value)
    def concatenate(self, value):
        """
        If value is a DoubleVal, returns the conjunction of the two values. Else, returns none.

        :param value: the value to concatenate
        :return: the concatenated result
        """
        from bn.values.none_val import NoneVal
        if isinstance(value, DoubleVal):
            return DoubleVal(self._value + value.get_double())

        from bn.values.string_val import StringVal
        if isinstance(value, StringVal):
            from bn.values.value_factory import ValueFactory
            return ValueFactory.create(str(self) + ' ' + str(value))

        if isinstance(value, NoneVal):
            return self

        self.log.warning("cannot concatenate " + str(self) + " and " + value)
        raise ValueError()

    @dispatch()
    def get_double(self):
        """
        Returns the double value

        :return: the value
        """
        return self._value
