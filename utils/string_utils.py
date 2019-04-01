import logging
from collections import Collection

import regex as re
from multipledispatch import dispatch

dispatch_namespace = dict()

class StringUtils:
    """
    Various utilities for manipulating strings
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    _n_best_regex = re.compile(r'.*\(([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\).*')
    _delimiters = ',.!?:;()[] \t\n'

    @staticmethod
    @dispatch(float, namespace=dispatch_namespace)
    def get_short_form(value):
        """
        Returns the string version of the double up to a certain decimal point.

        :param value:the float
        :return: the string
        """
        value = float(value)
        rounded = str(round(value * 10000.0) / 10000.0)

        return rounded[:-2] if rounded.endswith('.0') else rounded

    @staticmethod
    @dispatch(str, namespace=dispatch_namespace)
    def get_html_rendering(str_val):
        """
        Returns a HTML-compatible rendering of the raw string provided as argument

        :param str_val: the raw string
        :return: the formatted string
        """
        str_val = str_val.replace('phi', '&phi;')
        str_val = str_val.replace('theta', '&theta;')
        str_val = str_val.replace('psi', '&psi;')

        searcher = re.compile("_\\{(\\p{Alnum}*?)\\}").search(str_val)
        if searcher:
            subscript = searcher.group(1)
            str_val = str_val.replace("_{%s}" % subscript, "<sub>%s</sub>" % subscript)

        searcher = re.compile("_(\\p{Alnum}*)").search(str_val)
        if searcher:
            subscript = searcher.group(1)
            str_val = str_val.replace("_%s" % subscript, "<sub>%s</sub>" % subscript)

        searcher = re.compile("\\^\\{(\\p{Alnum}*?)\\}").search(str_val)
        if searcher:
            subscript = searcher.group(1)
            str_val = str_val.replace("^{%s}" % subscript, "<sup>%s</sup>" % subscript)

        searcher = re.compile("\\^([\\w\\-\\^]+)").search(str_val)
        if searcher:
            subscript = searcher.group(1)
            str_val = str_val.replace("^%s" % subscript, "<sup>%s</sup>" % subscript)

        return str_val

    @staticmethod
    @dispatch(str, str, namespace=dispatch_namespace)
    def count_nr_occurences(str_val, char_val):
        """
        Returns the total number of occurrences of the character in the string.

        :param str_val: the string
        :param char_val: the string to search for
        :return: the number of occurrences
        """
        return str_val.count(char_val)

    @staticmethod
    @dispatch(str, namespace=dispatch_namespace)
    def check_form(str_val):
        """
        Checks the form of the string to ensure that all parentheses, braces and
        brackets are balanced. Logs warning messages if problems are detected.

        :param str_val: the string
        :return: true if the form is correct, false otherwise
        """
        brackets = [('(', ')'), ('[', ']'), ('{', '}')]
        left_idx = 0
        right_idx = 1
        stack = list()

        for char_val in str_val:
            for pair in brackets:
                if char_val == pair[left_idx]:
                    stack.append(char_val)
                elif char_val == pair[right_idx]:
                    if len(stack) == 0:
                        return False
                    elif stack.pop() != pair[left_idx]:
                        return False

        if len(stack) == 0:
            return True
        else:
            return False

    @staticmethod
    @dispatch(str, str, namespace=dispatch_namespace)
    def compare(str1, str2):
        """
        Performs a lexicographic comparison of the two identifiers. If there is a
        difference between the number of primes in the two identifiers, returns it.
        Else, returns +1 if id1.compareTo(id2) is higher than 0, and -1 otherwise.

        :param str1: the first identifier
        :param str2: the second identifier
        :return: the result of the comparison
        """
        # TODO: check whether it is a bug
        if "'" in str1 or "'" in str2:
            compared_val = (len(str2) - len(str2.replace("'", ""))) - (len(str1) - len(str1.replace("'", "")))
            if compared_val != 0:
                return compared_val

        return 1 if str1 < str2 else -1

    @staticmethod
    @dispatch(Collection, str, namespace=dispatch_namespace)
    def join(object_list, separator):
        """
        Joins the string elements into a single string where the elements are joined
        by a specific string.

        :param object_list: the object lists
        :param separator: the string used to join the object_list
        :return: the concatenated string
        """
        return separator.join([str(x) for x in object_list])

    @staticmethod
    @dispatch(str, namespace=dispatch_namespace)
    def get_table_from_input(str_val):
        """
        Returns a table with probabilities from the provided GUI input

        :param str_val: the raw text expressing the table
        :return: the string values together with their probabilities
        """
        table = dict()

        for sub_text in str_val.split(';'):
            searcher = StringUtils._n_best_regex.search(sub_text)

            if searcher:
                prob_str = searcher.group(1)
                table[sub_text.replace("(%s)" % prob_str, "").strip()] = float(prob_str)
            else:
                table[sub_text.strip()] = 1.

        return table

    @staticmethod
    @dispatch(str, namespace=dispatch_namespace)
    def is_delimiter(char_val):
        return char_val in StringUtils._delimiters

    @staticmethod
    @dispatch(str, int, int, namespace=dispatch_namespace)
    def is_delimited(str_val, start_idx, end_idx):
        if start_idx > 0:
            if not StringUtils.is_delimiter(str_val[start_idx - 1]):
                return False

        if end_idx < len(str_val):
            if not StringUtils.is_delimiter(str_val[end_idx]):
                return False

        return True

    @staticmethod
    @dispatch(str, str, namespace=dispatch_namespace)
    def count_occurences(str_val, pattern):
        """
        Counts the occurrences of a particular pattern in the string.

        :param str_val: the string to use
        :param pattern: the pattern to search for
        :return: the number of occurrences
        """
        last_idx = 0
        cnt = 0

        while True:
            try:
                last_idx = str_val.index(pattern, last_idx)
                cnt += 1
                last_idx += len(pattern)
            except ValueError:
                break

        return cnt

    @staticmethod
    @dispatch(str, namespace=dispatch_namespace)
    def escape(str_val):
        str_list = list()
        escape_char = "()[]?.!^"
        str_val = ' '.join(str_val.split())

        skip = False
        for idx, char_val in enumerate(str_val):
            if skip:
                continue

            if char_val in escape_char:
                str_list.append('\\' + char_val)
            elif char_val == '{' and str_val[idx+1] == '}':
                skip = True
                continue
            else:
                str_list.append(char_val)

        return ''.join(str_list)
