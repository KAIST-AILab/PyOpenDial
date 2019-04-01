from datastructs.assignment import Assignment
from datastructs.math_expression import MathExpression
from domains.rules.parameters.parameter import Parameter

import logging
from multipledispatch import dispatch


class FixedParameter(Parameter):
    """
    Representation of a parameter fixed to one single specific value.
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    def __init__(self, arg1):
        if isinstance(arg1, float):
            param = arg1
            """
            Constructs a fixed parameter with the given value.
    
            :param param: the parameter value
            """
            self.param = param

        else:
            raise NotImplementedError()

    @dispatch()
    def get_value(self):
        """
        Returns the parameter value

        :return: the value for the parameter
        """
        return self.param

    @dispatch(Assignment)
    def get_value(self, assignment):
        """
        Returns the parameter value, ignoring the input

        :return: the value for the parameter
        """
        return self.param


    @dispatch()
    def get_variables(self):
        """
        Returns an empty set

        :return: an empty set of distributions
        """
        return []

    def __str__(self):
        """
        Returns the parameter value as a string

        :return: the string
        """
        return str(self.param)

    __repr__ = __str__

    def __hash__(self):
        """
        Returns the hashcode for the fixed parameter

        :return: the hashcode
        """
        return 2 * hash(self.param)

    @dispatch()
    def get_expression(self):
        """
        Returns the mathematical expression representing the parameter

        :return: the expression
        """
        return MathExpression("" + self.param)
