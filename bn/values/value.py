import abc
import functools


@functools.total_ordering
class Value:
    """
    Generic class for a variable value. The value can be:
        compared to other values
        copied in a new value
        check if it contains a sub-value
        concatenated with another value.
    """
    __metaclass__ = abc.ABCMeta

    def __hash__(self):
        raise NotImplementedError()

    def __eq__(self, other):
        raise NotImplementedError()

    def __lt__(self, other):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __copy__(self):
        raise NotImplementedError()

    def __contains__(self, item):
        raise NotImplementedError()

    def get_sub_values(self):
        raise NotImplementedError()

    def concatenate(self, value):
        raise NotImplementedError()
