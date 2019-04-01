from datastructs.assignment import Assignment
from datastructs.math_expression import MathExpression
from domains.rules.parameters.fixed_parameter import FixedParameter
from domains.rules.parameters.parameter import Parameter

import logging
from multipledispatch import dispatch


class ComplexParameter(Parameter):
    """
    Representation of a complex parameter expression. The class uses the exp4j package
    to dynamically evaluate the result of the expression.
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    def __init__(self, arg1):
        if isinstance(arg1, MathExpression):
            expression = arg1
            """
            Constructs a new complex parameter with the given expression, assuming the
            list of parameters is provided as labels within the expression.
    
            :param expression: the expression
            """
            self.expression = expression

        else:
            raise NotImplementedError()

    @dispatch(Assignment)
    def get_value(self, param):
        """
        Returns the parameter value corresponding to the expression and the assignment
        of values to the unknown parameters.
        """
        try:
            result = self.expression.evaluate(param)
            return result if result else 0.
        except Exception as e:
            self.log.warning("cannot evaluate %s given %s: %s" % (str(self), param, e))
            return 0.0

    @dispatch(Assignment)
    def ground(self, param):
        """
        Grounds the parameter by assigning the values in the assignment to the unknown variables

        :param param: the grounding assignment
        :return: the grounded parameter
        """
        if param.contains_vars(self.expression.get_variables()):
            try:
                result = self.expression.evaluate(param)
                return FixedParameter(result)
            except Exception as e:
                self.log.warning("cannot ground %s with %s" % (self.expression, param))
        filled = str(self.expression)
        for u in param.get_variables():
            filled = filled.replace(u, str(param.get_value(u)))
        return ComplexParameter(MathExpression(filled))

    @dispatch()
    def get_variables(self):
        """
        Returns the list of unknown parameter variables
        """
        return self.expression.get_variables()

    @dispatch()
    def get_expression(self):
        """
        Returns the mathematical expression representing the parameter

        :return: the expression
        """
        return self.expression

    def __str__(self):
        """
        Returns the parameter template as a string

        :return: the string
        """
        return str(self.expression)

    def __hash__(self):
        """
        Returns the hashcode for the parameter

        :return: the hashcode
        """
        return -3 * hash(self.expression)
