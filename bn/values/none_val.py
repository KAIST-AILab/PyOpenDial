from bn.values.value import Value
from multipledispatch import dispatch


class NoneVal(Value):
    """
    "None" value (describing the lack of value, or an empty assignment).
    """
    def __hash__(self):
        """
        Returns a hashcode for the value.

        :return: the hashcode
        """
        return 346

    def __eq__(self, other):
        """
        Returns true if both values are none.

        :param other: the object to compare
        :return: true if equals, false otherwise
        """
        return isinstance(other, NoneVal)

    def __lt__(self, other):
        """
        Compares the none value to another value.

        :param other: the object to compare
        :return: hashcode difference
        """
        # TODO: check refactor > Is this a proper way to compare?
        return hash(self) < hash(other)

    def __str__(self):
        """
        Returns the string "None".

        :return: the string
        """
        return "None"

    def __len__(self):
        """
        Returns 0.

        :return: 0
        """
        return 0

    def __copy__(self):
        """
        Returns its own instance.

        :return: its own instance
        """
        return self

    def __contains__(self, item):
        """
        True if subvalue is contained in the current instance, and false otherwise.

        :param item: subvalue the possibly contained value
        :return: true if contained, false otherwise
        """
        return False

    @dispatch(Value)
    def concatenate(self, value):
        """
        Returns the value value provided as argument.

        :param value: the value to concatenate
        :return: the concatenated result
        """
        return value

    @dispatch()
    def get_sub_values(self):
        """
        Returns an empty list.

        :return: the empty list
        """
        return []
