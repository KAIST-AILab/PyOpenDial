from bn.values.value import Value
import logging
from multipledispatch import dispatch


class BooleanVal(Value):
    """
    Representation of a boolean value.
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    def __init__(self, value):
        if isinstance(value, bool):
            """
            Creates the boolean value (protected, use the ValueFactory to create it).

            :param value: the boolean
            """
            self._value = value
        else:
            raise NotImplementedError()

    def __hash__(self):
        """
        Returns the hashcode of the boolean.

        :return: the hashcode
        """
        return -145 if self._value else +78

    def __eq__(self, other):
        """
        Returns true if the boolean value is similar, false otherwise.

        :param other: the value to compare
        :return: true if similar, false otherwise
        """
        if not isinstance(other, BooleanVal):
            return False

        return self._value == other.get_boolean()

    def __str__(self):
        """
        Returns a string representation of the boolean value.

        :return: the string representation of the boolean value
        """
        return str(self._value)

    def __lt__(self, other):
        """
        Compares the boolean to another value.

        :param other: the value to compare
        :return: usual ordering, or hashcode difference if the value is not a boolean
        """
        # TODO: check refactor > Is this a proper way to handle error case?
        if not isinstance(other, BooleanVal):
            return hash(self) < hash(other)

        return self._value < other.get_boolean()

    def __len__(self):
        """
        Returns 1.

        :return: 1
        """
        return 1

    def __copy__(self):
        """
        Copies the boolean value.

        :return: the copy
        """
        return BooleanVal(self._value)

    def __contains__(self, item):
        """
        Returns false.

        :param item: the value to check
        :return: false
        """
        return False

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
        If value is a BooleanVal, returns the conjunction of the two values. Else, returns none.

        :param value: the value to concatenate
        :return: the concatenated result
        """
        if isinstance(value, BooleanVal):
            return BooleanVal(self._value & value.get_boolean())

        from bn.values.none_val import NoneVal
        if isinstance(value, NoneVal):
            return self
        else:
            self.log.warning("cannot concatenate " + str(self) + " and " + value)
            raise ValueError()

    @dispatch()
    def get_boolean(self):
        """
        Returns the boolean value.

        :return: the boolean value
        """
        return self._value
