from collections import Collection

from bn.values.value import Value

import logging
from multipledispatch import dispatch


class SetVal(Value):
    """
    Value that is defined as a set of values.
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    def __init__(self, values):
        if isinstance(values, Collection) or isinstance(values, list):
            """
            Creates the set of values (protected, should be created via ValueFactory).

            :param values: the values
            """
            self._value = set()

            for value in values:
                if isinstance(value, SetVal):
                    self._value.update(value.get_sub_values())
                else:
                    self._value.add(value)
        else:
            raise NotImplementedError()

    def __hash__(self):
        """
        Returns the hashcode for the list.

        :return: the hashcode
        """
        return hash(frozenset(self._value))

    def __eq__(self, other):
        """
        Returns true if the lists are equals (contain the same elements), false otherwise.

        :param other: the object to compare
        :return: true if equal, false otherwise
        """
        if not isinstance(other, SetVal):
            return False

        return self._value == other.get_sub_values()

    def __len__(self):
        """
        Returns the set length.

        :return: the length
        """
        return len(self._value)

    def __str__(self):
        """
        Returns a string representation of the set.

        :return: the string
        """
        results = []
        for value in self._value:
            results.append(str(value))

        return '[%s]' % ', '.join(results)

    def __lt__(self, other):
        """
        Compares the list value to another value.

        :param other: the object to compare
        :return: hashcode difference
        """
        # TODO: check refactor > Is this a proper way to compare?
        return hash(self) < hash(other)

    def __copy__(self):
        """
        Returns a copy of the list.

        :return: the copy
        """
        return SetVal(self._value)

    def __contains__(self, item):
        """
        Returns true if subvalue is contained, and false otherwise.

        :param item: the value
        :return: true if contained, false otherwise
        """
        return item in self._value

    @dispatch()
    def get_sub_values(self):
        """
        Returns the set of values.

        :return: the set
        """
        return self._value

    @dispatch(Value)
    def concatenate(self, value):
        """
        Concatenates the two sets.

        :param value: the value
        :return: the concatenation of the two sets
        """
        if isinstance(value, SetVal):
            new_set = set()
            new_set.update(self._value)
            new_set.update(value.get_set_values())
            return SetVal(new_set)

        from bn.values.none_val import NoneVal
        if isinstance(value, NoneVal):
            return self

        new_set = set()
        new_set.update(self._value)
        new_set.add(value)
        return SetVal(new_set)

    @dispatch()
    def is_empty(self):
        """
        Returns true if the set is empty, and false otherwise.

        :return: true if the set is empty, and false otherwise
        """
        return len(self) == 0
