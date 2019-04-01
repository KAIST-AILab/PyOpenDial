import logging

from multipledispatch import dispatch

from bn.values.value_factory import ValueFactory
from datastructs.assignment import Assignment
from datastructs.math_expression import MathExpression
from templates.regex_template import RegexTemplate
from utils.string_utils import StringUtils

dispatch_namespace = dict()


class ArithmeticTemplate(RegexTemplate):
    # logger
    log = logging.getLogger('PyOpenDial')

    def __init__(self, str_val):
        if not isinstance(str_val, str):
            raise NotImplementedError("UNDEFINED PARAMETERS")

        super(ArithmeticTemplate, self).__init__(str_val)

    @dispatch(Assignment)
    def fill_slots(self, assignment):
        """
        Fills the slots of the template, and returns the result of the function
        evaluation. If the function is not a simple arithmetic expression,
        """
        filled = super(ArithmeticTemplate, self).fill_slots(assignment)
        if '{' in filled:
            return filled

        if ArithmeticTemplate.is_arithmetic_expression(filled):
            try:
                return StringUtils.get_short_form(MathExpression(filled).evaluate())
            # TODO: need to check exception handling
            except Exception as e:
                self.log.warning("cannot evaluate " + filled)
                return filled

        # handling expressions that manipulate sets
        # (using + and - to respectively add/remove elements)
        merge = ValueFactory.none()
        for str_val in filled.split("+"):
            negations = str_val.split("-")
            merge = merge.concatenate(ValueFactory.create(negations[0]))
            for negation in negations[1:]:
                values = merge.get_sub_values()

                old_value = ValueFactory.create(negation)
                if old_value in values:
                    values.remove(ValueFactory.create(negation))

                merge = ValueFactory.create(values)

        return str(merge)

    @staticmethod
    @dispatch(str, namespace=dispatch_namespace)
    def is_arithmetic_expression(exp):
        """
        Returns true if the string corresponds to an arithmetic expression, and false
        otherwise

        :param exp: the string to check
        :return: true if the string is an arithmetic expression, false otherwise
        """
        is_arithmetic_expression = False
        cur_str = ''
        for char_val in exp:
            if char_val == '+' or char_val == '-' or char_val == '/' or (char_val == '*' and len(exp) > 2):
                is_arithmetic_expression = True

            if char_val == '?' or char_val == '|' or char_val == '[' or char_val == '_' or char_val == "'":
                return False

            if char_val.isalpha():
                cur_str += char_val
                continue

            if StringUtils.is_delimiter(char_val):
                if cur_str not in MathExpression.fixed_functions:
                    return False

                is_arithmetic_expression = True
                cur_str = ''

        return is_arithmetic_expression
