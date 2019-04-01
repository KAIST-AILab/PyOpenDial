from bn.values.array_val import ArrayVal
from bn.values.double_val import DoubleVal
from datastructs.assignment import Assignment
from datastructs.math_expression import MathExpression
from domains.rules.parameters.parameter import Parameter

import logging
from multipledispatch import dispatch


class SingleParameter(Parameter):
    """
    Parameter represented by a single distribution over a continuous variable. If the
    variable is multivariate, the parameter represents a specific dimension of the
    multivariate distribution.
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    def __init__(self, arg1, arg2=None):
        if isinstance(arg1, str) and arg2 is None:
            param_id = arg1
            """
            Creates a new stochastic parameter for a univariate distribution.
    
            :param param_id: the parameter identifier
            """
            self.param_id = param_id
            self.dimension = -1

        elif isinstance(arg1, str) and isinstance(arg2, int):
            param_id, dimension = arg1, arg2
            """
            Creates a new stochastic parameter for a particular dimension of a
            multivariate distribution.
    
            :param param_id: the parameter identifier
            :param dimension: the dimension for the multivariate variable
            """
            self.param_id = param_id
            self.dimension = dimension

        else:
            raise NotImplementedError()

    @dispatch()
    def get_variables(self):
        """
        Returns a singleton with the parameter label

        :return: a collection with one element: the parameter distribution
        """
        return [self.param_id]

    @dispatch(Assignment)
    def get_value(self, assignment):
        """
        Returns the actual value for the parameter, as given in the input assignment
        (as a DoubleVal or ArrayVal). If the value is not given, throws an exception.

        :param assignment: the input assignment
        :return: the actual value for the parameter
        """
        value = assignment.get_value(self.param_id)

        if assignment.contains_var(self.param_id):
            if isinstance(value, DoubleVal):
                return (assignment.get_value(self.param_id)).get_double()
            elif isinstance(value, ArrayVal) and len(value.get_array()) > self.dimension:
                return value.get_array()[self.dimension]

        else:
            self.log.warning("input %s does not contain  %s" % (assignment, self.param_id))
            return 0.0

    @dispatch()
    def get_expression(self):
        """
        Returns the mathematical expression representing the parameter

        :return: the expression
        """
        return MathExpression(str(self))

    def __str__(self):
        """
        Returns a string representation of the stochastic parameter
        """
        if self.dimension != -1:
            return str(self.param_id) + "[" + str(self.dimension) + "]"
        else:
            return str(self.param_id)

    def __hash__(self):
        """
        Returns the hashcode for the parameter
        """
        return -hash(self.param_id) + self.dimension
