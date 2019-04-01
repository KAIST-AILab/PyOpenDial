from multipledispatch import dispatch
from datastructs.assignment import Assignment
from templates.template import Template, MatchResult
from utils.string_utils import StringUtils


class StringTemplate(Template):
    """
    Template for a string without any underspecified or optional elements. In other
    words, the match, find can be simplified to the usual string matching methods. The
    fillSlots method returns the string.
    """

    def __init__(self, str_val):
        if not isinstance(str_val, str):
            raise NotImplementedError("UNDEFINED PARAMETERS")
        """
        Creates a new string template.

        :param str_val: the string object
        """
        # the string corresponding to the template
        self._str_val = str_val
        # whether the string represents a whole word or phrase (and not a punctuation)
        self._whole = len(str_val) != 1 or not StringUtils.is_delimiter(str_val[0])
        # empty set of slots
        self._slots = set()

    @dispatch()
    def get_slots(self):
        """
        Returns an empty set.
        """
        return self._slots

    @dispatch()
    def is_under_specified(self):
        """
        Returns false
        """
        return False

    @dispatch(str)
    def match(self, str_val):
        """
        Returns a match result if the provided value is identical to the string
        template. Else, returns an unmatched result.
        """
        str_val = str_val.strip()

        if str_val.lower() == self._str_val.lower():
            return MatchResult(0, len(str_val))
        else:
            return MatchResult(False)

    @dispatch(str, int)
    def find(self, str_val, max_results):
        """
        Searches for all possible occurrences of the template in the provided string.
        Stops if the maximum number of results is reached.
        """
        str_val = str_val.strip()
        results = []

        start = 0
        while True:
            try:
                start = str_val.index(self._str_val, start)
                end = start + len(self._str_val)
                if not self._whole or StringUtils.is_delimited(str_val, start, end):
                    results.append(MatchResult(start, end))
                if len(results) >= max_results:
                    break
                start = end
            except ValueError:
                break

        return results

    @dispatch(Assignment)
    def is_filled_by(self, input_val):
        """
        Returns true
        """
        return True

    @dispatch(Assignment)
    def fill_slots(self, fillers):
        """
        Returns the string itself
        """
        return self._str_val

    def __hash__(self):
        """
        Returns the hashcode for the string
        """
        return hash(self._str_val)

    def __str__(self):
        """
        Returns the string itself
        """
        return self._str_val

    def __eq__(self, other):
        """
        Returns true if the object is an identical string template
        """
        if not isinstance(other, StringTemplate):
            return False

        return self._str_val == other._str_val

    def __lt__(self, other):
        if not isinstance(other, StringTemplate):
            return False

        return self._str_val < other._str_val
