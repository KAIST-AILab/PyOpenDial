import logging
import math
import random
from collections import Collection, Callable

from multipledispatch import dispatch

from settings import Settings


class Intervals:
    """
    * Representation of a collection of intervals, each of which is associated with a
    * content object, and start and end values. The difference between the start and end
    * values of an interval can for instance represent the object probability.
    *
    * The intervals can then be used for sampling a content object according to the
    * defined intervals.
    """
    log = logging.getLogger('PyOpenDial')

    def __init__(self, arg1=None, arg2=None):
        if isinstance(arg1, dict) and arg2 is None:
            table = arg1
            """
            Creates a new interval collection with a set of (content,probability) pairs

            :param table: the tables from which to create the intervals could not be
                          created
            """
            self._intervals = list()
            self._total_prob = 0.

            for key in table.keys():
                prob = table.get(key)
                if math.isnan(prob):
                    raise ValueError('probability is NaN: ' + str(table))

                self._intervals.append(Interval(key, self._total_prob, self._total_prob + prob))
                self._total_prob += prob

            if self._total_prob < Settings.eps:
                raise ValueError('total prob is null: ' + str(self._total_prob))
        elif isinstance(arg1, Collection) and isinstance(arg2, Callable):
            contents = arg1
            probs = arg2
            """
            Creates a new interval collection with a collection of values and a function
            specifying the probability of each value

            :param contents: the collection of content objects
            :param probs: the function associating a weight to each object intervals could
                        not be created
            """
            self._intervals = list()
            self._total_prob = 0.

            for content in contents:
                prob = probs(content)
                if math.isnan(prob):
                    raise ValueError('probability is Nan: ' + str(content))

                self._intervals.append(Interval(content, self._total_prob, self._total_prob + prob))
                self._total_prob += prob

            if self._total_prob < Settings.eps:
                raise ValueError('total prob is null: ' + str(self._total_prob))
        else:
            raise NotImplementedError("UNDEFINED PARAMETERS")

    def sample(self):
        """
        Samples an object from the interval collection, using a simple binary search
        procedure.

        :return: the sampled object
        """
        if len(self._intervals) == 0:
            raise ValueError('could not sample: empty interval')

        rand = random.random() * self._total_prob

        min_idx = 0
        max_idx = len(self._intervals)
        while min_idx <= max_idx:
            mid = int(min_idx + (max_idx - min_idx) / 2)
            compare_value = self._intervals[mid].compare_to(rand)
            if compare_value > 0:
                max_idx = mid - 1
            elif compare_value == 0:
                return self._intervals[mid].get_content()
            elif compare_value < 0:
                min_idx = mid + 1

        raise ValueError("could not sample given the intervals: " + self.__str__())

    def __str__(self):
        return '\n'.join([str(s) for s in self._intervals])

    def __len__(self):
        return len(self._intervals)

    def is_empty(self):
        """
        Returns true is the interval is empty (no elements), false otherwise

        :return: whether the interval is empty
        """
        return len(self._intervals) == 0


class Interval:
    """
    Representation of a single interval, made of an object, a start value, and an end
    value
    """

    def __init__(self, content, start, end):
        if not isinstance(content, object) or not isinstance(start, float) or not isinstance(end, float):
            raise NotImplementedError("UNDEFINED PARAMETERS")

        """
        Creates a new interval

        :param content: the interval content
        :param start: the start value for the interval
        :param end: the end value for the interval
        """
        self._content = content
        self._start = start
        self._end = end

    def get_content(self):
        """
        Returns the object associated with the interval

        :return: the object
        """
        return self._content

    def __str__(self):
        return '%s[%f, %f]' % (str(self._content), self._start, self._end)

    @dispatch(float)
    def compare_to(self, value):
        """
        Returns the position of the double value in comparison with the start and end
        values of the interval

        :param value: the value to compare
        :return +1 if value < start, 0 if start <= value <= end, -1 if value >= end
        """
        if value >= self._start:
            if value < self._end:
                return 0
            else:
                return -1

        return 1
