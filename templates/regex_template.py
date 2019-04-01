from multipledispatch import dispatch
import regex as re
import logging

from bn.values.custom_val import CustomVal
from bn.values.value_factory import ValueFactory
from datastructs.assignment import Assignment
from templates.template import Template, MatchResult
from utils.string_utils import StringUtils

dispatch_namespace = dict()


class RegexTemplate(Template):
    """
    Template based on regular expressions. Syntax for the templates:
    - underspecified slots, represented with braces, e.g. {Slot}
    - optional elements, surrounded by parentheses followed by a question mark, e.g. (option)?
    - alternative elements, surrounded by parentheses and separated by the | character, i.e. (option1|option2)
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    _slot_regex = re.compile("\\{(.+?)\\}")
    _alt_regex = re.compile("(\\\\\\(((\\(\\?)|[^\\(])+?\\\\\\)\\\\\\?)"
                            + "|(\\\\\\(((\\(\\?)|[^\\(])+?\\|((\\(\\?)"
                            + "|[^\\(])+?\\\\\\)(\\\\\\?)?)")

    def __init__(self, str_val):
        if not isinstance(str_val, str):
            raise NotImplementedError("UNDEFINED PARAMETERS")
        """
        Constructs the regular expression, based on the string representation.

        :param str_val: the string
        """
        self._str_val = str_val.strip()

        escaped_str_val = StringUtils.escape(str_val)
        regex = RegexTemplate._construct_regex(escaped_str_val)

        self._pattern = re.compile(regex, re.I | re.U)
        self._slots = self.get_slots(str_val)

    @staticmethod
    @dispatch(str, namespace=dispatch_namespace)
    def _is_possible_regex(str_val):
        """
        Checks whether the string could possibly represent a regular expression (this
        is just a first, fast guess, which will need to be verified by actually
        constructing the regex using the constructRegex method below).

        :param str_val: the string
        :return: true if the string is likely to be a regular expression, else false
        """
        # TODO: check whether this is a bug or not.
        for idx, char_val in enumerate(str_val):
            if char_val == '*':
                return True
            if char_val == '{':
                if idx < len(str_val) - 1 and str_val[idx + 1] != '}':
                    return True
            if char_val == '|':
                return True
            if char_val == '?':
                if idx > 1 and str_val[idx - 1] == ')':
                    return True

        return False

    @dispatch(namespace=dispatch_namespace)
    def get_slots(self):
        """
        Returns the (possibly empty) set of slots for the template
        """
        return set(self._slots.keys())

    @dispatch(str, namespace=dispatch_namespace)
    def match(self, str_val):
        """
        Tries to match the template against the provided string.
        """
        input = str_val.strip()

        matcher = self._pattern.fullmatch(input)
        if matcher:
            results = MatchResult(matcher.start(), matcher.end())
            for slot_key in self._slots.keys():
                filled_value = matcher.captures(self._slots[slot_key])[0]
                if not StringUtils.check_form(filled_value) and self.permutate_pattern():
                    return self.match(str_val)
                results.add_pair(slot_key, filled_value)

            return results

        return MatchResult(False)

    @dispatch(namespace=dispatch_namespace)
    def is_under_specified(self):
        """
        Returns true.
        """
        return True

    @dispatch(str, int, namespace=dispatch_namespace)
    def find(self, str_val, max_results):
        """
        Tries to find all occurrences of the template in the provided string. Stops
        after the maximum number of results is reached.
        """
        str_val = str_val.strip()
        results = list()

        for matcher in self._pattern.finditer(str_val):
            if not StringUtils.is_delimited(str_val, matcher.start(), matcher.end()):
                continue

            match_result = MatchResult(matcher.start(), matcher.end())
            for slot_key in self._slots.keys():
                filled_value = matcher.group(self._slots[slot_key]).strip()

                # quick-fix to handle some rare cases where the occurrence found
                # by the regex leads to unbalanced parentheses or brackets.
                # TODO: check whether this is a bug or not.
                if not StringUtils.check_form(filled_value) and self.permutate_pattern():
                    return self.find(str_val, max_results)

                match_result.add_pair(slot_key, filled_value)

            results.append(match_result)

            if len(results) >= max_results:
                break

        return results

    @dispatch(Assignment, namespace=dispatch_namespace)
    def is_filled_by(self, assignment):
        """
        Returns true if all slots are filled by the assignment. Else, returns false.
        """
        # TODO: check whether this function has a bug or not.
        for slot_key in self._slots:
            value = assignment.get_value(slot_key)
            if value == ValueFactory.none():
                return False

        return True

    @dispatch(Assignment, namespace=dispatch_namespace)
    def fill_slots(self, fillers):
        """
        Fills the template with the given content, and returns the filled string. The
        content provided in the form of a slot:filler mapping. For instance, given a
        template: "my name is {name}" and a filler "name:Pierre", the method will
        return "my name is Pierre".

        :param fillers: the content associated with each slot.
        :return: the string filled with the given content
        """
        if len(self._slots) == 0:
            return self._str_val

        result = self._str_val
        for slot_key in self._slots.keys():
            value = fillers.get_value(slot_key)
            if isinstance(value, CustomVal):
                assert("{%s}" % slot_key == self._str_val)  # only {custom_value} is allowed.
                result = value.get_value()
            elif value != ValueFactory.none():
                str_val = str(value)
                result = result.replace("{%s}" % slot_key, str_val)

        return result

    def __hash__(self):
        """
        Returns the hashcode for the raw string.
        """
        return hash(self._str_val)

    def __str__(self):
        """
        Returns the raw string.
        """
        return self._str_val

    def __eq__(self, other):
        """
        Returns true if o is a RegexTemplate with the same string. Else false.
        """
        if not isinstance(other, RegexTemplate):
            return False

        return self._str_val == other._str_val

    @dispatch(namespace=dispatch_namespace)
    def permutate_pattern(self):
        """
        Quick fix to make slight changes to the regular expression in case the
        templates produces matching results with unbalanced parenthesis/brackets. For
        instance, when the template pred({X},{Y}) is matched against a string
        pred(foo,bar(1,2)), the resulting match is X="foo,bar(1" and Y="2)". We can
        get the desired result X="foo", Y="bar(1,2)" by changing the patterns,
        replacing greedy quantifiers by reluctant or possessive ones.

        :return: true if the permutation resulted in a change in the pattern. else, false.
        """
        new_pattern = self._pattern.pattern.replace("(.+)", "(.+?)", 1)
        if new_pattern == self._pattern.pattern:
            new_pattern = self._pattern.sub("(.?)", "(.++)", 1)

        change = new_pattern != self._pattern.pattern
        self._pattern = re.compile(new_pattern, re.I | re.U)
        return change

    @staticmethod
    @dispatch(str, namespace=dispatch_namespace)
    def _construct_regex(init):
        """
        Formats the regular expression corresponding to the provided string

        :param init: the initial string
        :return: the corresponding expression
        """
        has_stars = False
        has_slots = False
        has_alternatives = False

        for char_val in init:
            if char_val == '*':
                has_stars = True
            elif char_val == '{':
                has_slots = True
            elif char_val == '|' or char_val == '?':
                has_alternatives = True

        result = RegexTemplate._replace_stars(init) if has_stars else init
        result = RegexTemplate._slot_regex.sub((lambda x: ('(.+)')), result) if has_slots else result
        result = RegexTemplate._replace_complex(result) if has_alternatives else result

        return result

    @dispatch(str)
    def get_slots(self, str_val):
        """
        Returns the slots defined in the string as well as their sequential order
        (starting at 1) in the string.

        :param str_val: the string to analyse
        :return: the extracted slots
        """
        slots = dict()
        for m in RegexTemplate._slot_regex.finditer(str_val):
            variable = m.group(1)
            if variable not in slots:
                slots[variable] = len(slots) + 1

        return slots


    @staticmethod
    @dispatch(str, namespace=dispatch_namespace)
    def _replace_stars(str_val):
        """
        Replaces the * characters in the string by a proper regular expression

        :param str_val: the initial string
        :return: the formatted expression
        """
        results = []

        skip_iteration = False
        for idx, char_val in enumerate(str_val):
            if skip_iteration:
                skip_iteration = False
                continue

            # TODO: check whether the condition below has a bug or not.
            if char_val == '*' and idx == 0 and len(str_val) > 1 and str_val[idx + 1] == ' ':
                results.append("(?:.+ |)")
                skip_iteration = True
            elif char_val == '*' and 0 < idx < len(str_val) - 1 and str_val[idx + 1] == ' ' and str_val[idx - 1] == ' ':
                results[len(results) - 1] = results[len(results) - 1][:-1]
                results.append("(?:.+|)")
            elif char_val == '*' and 0 < idx == len(str_val) - 1 and str_val[idx - 1] == ' ':
                results[len(results) - 1] = results[len(results) - 1][:-1]
                results.append("(?: .+|)")
            elif char_val == '*':
                results.append("(?:.*)")
            else:
                results.append(char_val)

        return ''.join(results)

    @staticmethod
    @dispatch(str, namespace=dispatch_namespace)
    def _replace_complex(str_val):
        """
        Replace the alternative or optional elements by a proper regular expression

        :param str_val: the initial string
        :return: the formatted expression
        """
        result = str_val
        matcher = RegexTemplate._alt_regex.search(result)
        while matcher:
            group_val = matcher.captures()[0]
            if not StringUtils.check_form(group_val):
                continue

            if group_val.endswith('?'):
                core = group_val[2:len(group_val) - 4]
                if matcher.end() < len(result) and result[matcher.end()] == ' ':
                    result = result[:matcher.start()] + ("(?:" + core.replace("\\|", " \\|") + " )?") + result[matcher.end() + 1:]
                elif matcher.end() >= len(result) and matcher.start() > 0 and result[matcher.start() - 1] == ' ':
                    result = result[:matcher.start() - 1] + ("(?: " + core.replace("\\|", " \\|") + ")?") + result[matcher.end():]
                else:
                    result = result[:matcher.start()] + "(?:" + core + ")?" + result[matcher.end():]
            else:
                core = group_val[2:len(group_val) - 2]
                result = result[:matcher.start()] + "(?:" + core + ")" + result[matcher.end():]

            matcher = RegexTemplate._alt_regex.search(result)

        return result
