from bn.values.value import Value

import logging
from multipledispatch import dispatch


class StringVal(Value):
    """
    String value.
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    def __init__(self, value):
        if isinstance(value, str):
            """
            Creates a new string value (protected, use the ValueFactory instead).

            :param value: the string
            """
            self._value = value
            self._template = None
        else:
            raise NotImplementedError()

    def __hash__(self):
        """
        Returns the hashcode for the string.

        :return: the hashcode
        """
        return hash(self._value.lower())

    def __eq__(self, other):
        """
        Returns true if the strings are equals, false otherwise.

        :param other: the object to compare
        :return: true if equals, false otherwise
        """
        if not isinstance(other, StringVal):
            return False

        return self._value.lower() == str(other).lower()

    def __lt__(self, other):
        """
        Compares the string value to another value.

        :param other: the object to compare
        :return: usual ordering, or hashcode if the value is not a string
        """
        if not isinstance(other, StringVal):
            return False

        return self._value < str(other)

    def __str__(self):
        """
        Returns the string itself.

        :return: the string
        """
        return self._value

    def __len__(self):
        """
        Returns the string length.

        :return: the length
        """
        return len(self._value)

    def __copy__(self):
        """
        Returns a copy of the string value.

        :return: the copy
        """
        return StringVal(self._value)

    def __contains__(self, item):
        """
        Returns true if subvalue is a substring of the current StringVal, and false otherwise.

        :param item: the value
        :return: true is subvalue is a substring of the object, false otherwise
        """
        if isinstance(item, StringVal):
            if item._template is None:
                from templates.template import Template
                item._template = Template.create(item._value)

            return item._template.partial_match(self._value).is_matching()
        else:
            # TODO: check bug > subvalue.toString().contains(str): not a proper comparison
            return str(item) in self._value  # corrected version.

    @dispatch()
    def get_sub_values(self):
        """
        Returns a list of words.

        :return: list of words
        """
        from bn.values.value_factory import ValueFactory
        return [ValueFactory.create(w) for w in self._value.split(" ")]

    @dispatch(Value)
    def concatenate(self, value):
        """
        Returns the concatenation of the two values.

        :param value: the value
        :return: the concatenation of the two values
        """
        if isinstance(value, StringVal):
            from bn.values.value_factory import ValueFactory
            return ValueFactory.create(self._value + " " + str(value))

        from bn.values.double_val import DoubleVal
        if isinstance(value, DoubleVal):
            from bn.values.value_factory import ValueFactory
            return ValueFactory.create(self._value + " " + str(value))

        from bn.values.none_val import NoneVal
        if isinstance(value, NoneVal):
            return self

        self.log.warning("cannot concatenate " + str(self) + " and " + value)
        raise ValueError()
