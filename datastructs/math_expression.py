import logging
from copy import copy

import regex as re
from asteval import Interpreter
from multipledispatch import dispatch

from bn.values.array_val import ArrayVal
from bn.values.double_val import DoubleVal
from datastructs.assignment import Assignment
from settings import Settings
from templates.template import Template

dispatch_namespace = dict()


def expr(expression, context=None):
    if context is None:
        context = dict()

    interpreter = Interpreter()
    for variable, value in context.items():
        interpreter.symtable[variable] = value

    return interpreter(expression)


class MathExpressionWrapper:
    pass


class MathExpression(MathExpressionWrapper):
    """
    Representation of a mathematical expression whose value can be evaluated. The
    expression may contain unknown variables. In this case, one can evaluate the value
    of the expression given a particular assignment of values.
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    varlabel_regex = re.compile(r'([a-zA-Z][\w_\.]*)')

    fixed_functions = ["abs", "acos", "asin", "atan", "cbrt", "ceil", "cos", "cosh", "exp", "floor", "log", "log10",
                       "log2", "sin", "sinh", "sqrt", "tan", "tanh"]

    function_pattern = re.compile(r"\w+\(")

    def __init__(self, arg1):
        if isinstance(arg1, str):
            expression = arg1
            """
            Creates a new mathematical expression from the string
    
            :param value: the expression
            """
            self._raw_expression_str = expression
            self._functions = MathExpression.get_functions(expression)
            self._variables = set()

            local = ','.join([x.strip() for x in expression.split(',')])
            for functional_template in self._functions:
                self._variables.update(functional_template.get_slots())
                local = local.replace(str(functional_template), functional_template.get_function().__name__ + str(hash(functional_template)))

            self._variables.update(self.get_variable_labels(local))
            for functional_template in self._functions:
                self._variables.remove(functional_template.get_function().__name__ + str(hash(functional_template)))

            # expression = re.sub(r'[\^]', '**', expression)
            local = re.sub(r'[\[\]\{\}]', '', local)
            local = re.sub(r'\.([a-zA-Z])', '_$1', local)
            self._expression_str = local
        elif isinstance(arg1, MathExpressionWrapper):
            existing = arg1
            """
            Creates a new mathematical expression that is a copy from another one
    
            :param value: existing the expression to copy
            """
            self._raw_expression_str = existing._raw_expression_str
            self._expression_str = existing._expression_str
            self._variables = existing._variables
            self._functions = existing._functions

        else:
            raise NotImplementedError()

    @staticmethod
    @dispatch(str, namespace=dispatch_namespace)
    def get_functions(expression):
        functions = set()

        for matcher in MathExpression.function_pattern.finditer(expression):
            function_s = matcher.captures()[0]
            nb_open_parentheses = 0
            for idx, c in enumerate(expression):
                if c == '(':
                    nb_open_parentheses += 1
                elif c == ')' and nb_open_parentheses > 1:
                    nb_open_parentheses -= 1
                elif c == ')':
                    function_s += expression[matcher.end():idx+1]
                    if Settings.is_function(function_s):
                        functional_template = Template.create(function_s)
                        functions.add(functional_template)
                    break

        return functions

    @dispatch()
    def get_variables(self):
        """
        Returns the unknown variable labels in the expression

        :return: the variable labels
        """
        return self._variables

    @dispatch()
    def evaluate(self):
        """
        Evaluates the result of the expression

        :return: the result
        """
        if len(self._variables) > 0:
            raise ValueError()

        return float(expr(self._expression_str))

    @dispatch(Assignment)
    def evaluate(self, param):
        """
        Evaluates the result of the expression, given an assignment of values to the
        unknown variables

        :param param: the assignment
        :return: the result
        """
        param = param if len(self._functions) == 0 else copy(param)

        for functional_template in self._functions:
            custom_function = functional_template.get_function()
            result = functional_template.get_value(param)
            param.add_pair(custom_function.__name__ + str(hash(functional_template)), result)

        return expr(self._expression_str, self.get_doubles(param))

    @dispatch(str, list)
    def combine(self, operator, elements):
        """
        Combines the current expression with one or more other expressions and a
        binary operator (such as +,* or -).

        :param operator: the operator between the expression
        :param elements: the elements to add
        :return: the expression corresponding to the combination
        """
        expression_list = [self._raw_expression_str]
        expression_list.extend([element._raw_expression_str for element in elements])

        new_expression = '(' + operator.join(expression_list) + ')'
        return MathExpression(new_expression)

    @staticmethod
    @dispatch(str, namespace=dispatch_namespace)
    def get_variable_labels(str_val):
        """
        Returns a set of possible variable labels in the given string.

        :param str_val: the string to analyse
        :return: the extracted labels
        """
        indexed_vars = set()
        for matcher in MathExpression.varlabel_regex.finditer(str_val):
            var_label = matcher.captures()[0]
            if var_label not in MathExpression.fixed_functions:
                indexed_vars.add(var_label)

        return indexed_vars

    @staticmethod
    @dispatch(Assignment, namespace=dispatch_namespace)
    def get_doubles(assignment):
        """
        Returns a representation of the assignment limited to double values. For
        arrays, each variable is expanded into separate variables for each dimension.

        :param assignment: the assignment
        :return: the assignment of all double (and array) values
        """
        doubles = dict()

        for variable in assignment.get_variables():
            value = assignment.get_value(variable)
            if not (isinstance(value, DoubleVal) or isinstance(value, ArrayVal)):
                continue

            variable = re.sub(r'\.', '_', variable)

            if isinstance(value, DoubleVal):
                doubles[variable] = value.get_double()
            elif isinstance(value, ArrayVal):
                array = value.get_array()
                for idx, value in enumerate(array):
                    doubles[variable + str(idx)] = value

        return doubles

    def __str__(self):
        """
        Returns a string representation of the expression
        """
        return self._raw_expression_str

    def __eq__(self, other):
        """
        Returns true if the expressions are identical, false otherwise
        """
        return isinstance(other, MathExpression) and (self._expression_str == other._expression_str)

    def __hash__(self):
        """
        Returns the hashcode for the expression
        """
        return hash(self._expression_str)
