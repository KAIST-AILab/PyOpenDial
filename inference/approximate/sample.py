import logging
import math
from functools import total_ordering

from multipledispatch import dispatch

from datastructs.assignment import Assignment


@total_ordering
class Sample(Assignment):
    """
    Representation of a (possibly weighted) sample, which consists of an assignment of
    values together with a weight (here in logarithmic form) and utility.
    """
    log = logging.getLogger('PyOpenDial')

    def __init__(self, arg1=None, arg2=None, arg3=None):
        if arg1 is None and arg2 is None and arg3 is None:
            """
            Creates a new, empty weighted sample
            """
            super(Sample, self).__init__()
            self._log_weight = 0.
            self._utility = 0.
        elif isinstance(arg1, Assignment) and arg2 is None and arg3 is None:
            assignment = arg1
            """
            Creates a new sample
    
            :param assignment: the existing assignment
            """
            super(Sample, self).__init__(assignment)
            self._log_weight = 0.
            self._utility = 0.
        elif isinstance(arg1, Assignment) and isinstance(arg2, float) and isinstance(arg3, float):
            assignment = arg1
            log_weight = arg2
            utility = arg3
            """
            Creates a new sample with an existing weight and utility
    
            :param assignment: the assignment
            :param log_weight: the logarithmic weight
            :param utility: the utility
            """
            super(Sample, self).__init__(assignment)
            self._log_weight = log_weight
            self._utility = utility
        else:
            raise NotImplementedError("UNDEFINED PARAMETERS")

    @dispatch(float)
    def add_log_weight(self, log_weight):
        """
        Adds a logarithmic weight to the current one

        :param log_weight: the weight to add
        """
        self._log_weight += log_weight

    @dispatch(float)
    def set_log_weight(self, log_weight):
        """
        Sets the logarithmic weight to a given value

        :param log_weight: the value for the weight
        """
        self._log_weight = log_weight

    def get_weight(self):
        """
        Returns the sample weight (exponentiated value, not the logarithmic one!)

        :return: the (exponentiated) weight for the sample
        """
        return math.exp(self._log_weight)

    @dispatch(float)
    def set_weight(self, weight):
        """
        Sets the logarithmic weight to a given value

        :param weight: the value for the weight
        """
        self._log_weight = math.log(weight)

    @dispatch(float)
    def add_utility(self, utility):
        """
        Adds a utility to the sample

        :param utility: the utility to add
        """
        self._utility += utility

    def get_utility(self):
        """
        Returns the utility of the sample

        :return: the utility
        """
        return self._utility

    def __str__(self):
        return '%s(w=%f, util=%f)' % (super().__str__(), self.get_weight(), self._utility)

    def __lt__(self, other):
        return int((self._utility - other._utility) * 1000)
