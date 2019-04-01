import abc
import logging

from multipledispatch import dispatch

from datastructs.assignment import Assignment
from datastructs.graph import Graph
from settings import Settings

dispatch_namespace = dict()


class Template:
    """
    Representation of a (possibly underspecified) template for a Value. An example of
    template is "Foobar({X},{Y})", where {X} and {Y} are underspecified slots that can
    take any value. One can also use template to encode simple regular expressions
    such as "the (big)? (box|cylinder)", which can match strings such as "the box" or
    "the big cylinder".

    Template objects supports 3 core operations:
    - match(string): full match of the template against the string. If the template
      contains slots, returns their values resulting from the match.
    - find(string): find all possible occurrences of the template in the string.
    - fillSlots(assignment): returns the string resulting from the replacement of
      the slot values by their values in the assignment.

    Different implements of the template interface are provided, to allow for
    "full string" templates (without any underspecification), templates based on
    regular expressions, mathematical expressions, and templates operating on semantic
    graphs.
    """

    __metaclass__ = abc.ABCMeta

    # logger
    log = logging.getLogger('PyOpenDial')

    @staticmethod
    @dispatch(str, namespace=dispatch_namespace)
    def create(value):
        """
        Creates a new template based on the string value. This method finds the best
        template representation for the string and returns the result.

        :param value: the string for the template
        :return: the corresponding template object
        """

        if Settings.is_function(value):
            from templates.functional_template import FunctionalTemplate
            return FunctionalTemplate(value)

        try:
            from templates.arithmetic_template import ArithmeticTemplate
            if Graph.is_relational(value):
                from templates.relational_template import RelationalTemplate
                return RelationalTemplate(value)

            from templates.regex_template import RegexTemplate
            if RegexTemplate._is_possible_regex(value):
                if ArithmeticTemplate.is_arithmetic_expression(value):
                    return ArithmeticTemplate(value)
                else:
                    return RegexTemplate(value)

            from templates.string_template import StringTemplate
            return StringTemplate(value)
        except:
            from templates.string_template import StringTemplate
            return StringTemplate(value)

    @dispatch()
    @abc.abstractmethod
    def get_slots(self):
        """
        Returns the (possibly empty) set of slots for the template

        :return: the set of slots
        """
        raise NotImplementedError()

    @dispatch()
    @abc.abstractmethod
    def is_under_specified(self):
        """
        Returns true if the template is an actual template, i.e. can match multiple
        values (due to slots or alternative/optional elements)

        :return: true if underspecified, false otherwise
        """
        raise NotImplementedError()

    @dispatch(str)
    @abc.abstractmethod
    def match(self, str_val):
        """
        Checks whether the string is matching the template or not. The matching result
        contains a boolean representing the outcome of the process, as well (if the
        match is successful) as the boundaries of the match and the extracted slot
        values.

        :param str_val: the string to check
        :return: the matching result
        """
        raise NotImplementedError()

    @dispatch(str)
    def partial_match(self, str_val):
        """
        Checks whether the template can be found within the string. The matching
        result contains a boolean representing the outcome of the process, as well (if
        the match is successful) as the boundaries of the match and the extracted slot
        values.

        :param str_val: the string to check
        :return: the mathcing result
        """
        results = self.find(str_val, 1)
        if len(results) == 0:
            return MatchResult(False)
        else:
            return results[0]

    @dispatch(str, int)
    @abc.abstractmethod
    def find(self, str_val, max_results):
        """
        Searches for all occurrences of the template in the str. The maximum number of
        ccurrences to find is specified in maxResults.

        :param str_val: the string to check
        :param max_results: maxResults the maximum number of occurrences
        :return: the matching results
        """
        raise NotImplementedError()

    @dispatch(Assignment)
    @abc.abstractmethod
    def is_filled_by(self, input_val):
        """
        Returns true if the provided variables cover all of the slots in the template.
        Otherwise, returns false.

        :param input_val: the input
        :return: true if all slots can be filled, and false otherwise.
        """
        raise NotImplementedError()

    @dispatch(Assignment)
    @abc.abstractmethod
    def fill_slots(self, fillers):
        """
        Fills the template with the given content, and returns the filled string. The
        content provided in the form of a slot:filler mapping. For instance, given a
        template: "my name is {name}" and a filler "name:Pierre", the method will
        return "my name is Pierre".

        :param fillers: the content associated with each slot.
        :return: the string filled with the given content
        """
        raise NotImplementedError()

    def __eq__(self, other):
        """
        Compares the templates (based on their string value)
        """
        if not isinstance(other, Template):
            return False
        return str(self) == str(other)

    def __lt__(self, other):
        """
        Compares the templates (based on their string value)
        """
        if not isinstance(other, Template):
            return False
        return str(self) < str(other)


class MatchResult(Assignment):
    """
    Representation of a matching result
    """

    def __init__(self, arg1=None, arg2=None):
        if isinstance(arg1, bool) and arg2 is None:
            is_matching = arg1
            super().__init__()
            self._is_matching = is_matching
        elif isinstance(arg1, int) and isinstance(arg2, int):
            start = arg1
            end = arg2
            super().__init__()
            self._is_matching = True
        else:
            raise NotImplementedError("UNDEFINED PARAMETERS")

    @dispatch()
    def is_matching(self):
        return self._is_matching

    def __str__(self):
        return "%s (%s)" % (str(self._is_matching), super().__str__())

    __repr__ = __str__

    def __eq__(self, other):
        if not isinstance(other, MatchResult):
            return False

        return self._map == other._map and self._is_matching == other._is_matching

    def __hash__(self):
        return hash(self._is_matching) - super(Assignment, self).__hash__()

    def __copy__(self):
        match_result = MatchResult(self._is_matching)
        match_result.add_assignment(self)
        return match_result
