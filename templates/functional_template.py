from multipledispatch import dispatch
from bn.values.value_factory import ValueFactory
from datastructs.assignment import Assignment
from settings import Settings
from templates.string_template import StringTemplate
from templates.template import Template, MatchResult


class FunctionalTemplate(Template):

    def __init__(self, str_val):
        if not isinstance(str_val, str):
            raise NotImplementedError("UNDEFINED PARAMETERS")

        """
        Creates a new string template.

        :param str_val: the string object
        """

        function_name = str_val[0:str_val.index('(')]
        self._function = Settings.get_function(function_name)
        self._parameters = list()
        self._slots = set()

        cur_param = list()
        open_params = 0

        for char_val in str_val[str_val.index('(') + 1:-1]:
            if char_val == '(':
                open_params += 1
            elif char_val == ')':
                open_params -= 1
            elif open_params == 0 and char_val == ',':
                param = Template.create(''.join(cur_param))
                self._parameters.append(param)
                self._slots.update(param.get_slots())
                cur_param = list()
                continue

            cur_param.append(char_val)

        param = Template.create(''.join(cur_param))
        self._parameters.append(param)
        self._slots.update(param.get_slots())

    @dispatch()
    def get_slots(self):
        return self._slots

    @dispatch()
    def is_under_specified(self):
        return len(self._slots) != 0

    @dispatch(str)
    def match(self, str_val):
        if self.is_under_specified():
            return StringTemplate(self.fill_slots(Assignment())).match(str_val)

        return MatchResult(False)

    @dispatch(str, int)
    def find(self, str_val, max_results):
        if self.is_under_specified():
            return StringTemplate(self.fill_slots(Assignment())).find(str_val, max_results)

        return list()

    @dispatch(Assignment)
    def is_filled_by(self, input):
        return input.contains_vars(self._slots)

    @dispatch(Assignment)
    def fill_slots(self, fillers):
        return str(self.get_value(fillers))

    @dispatch(Assignment)
    def get_value(self, fillers):
        filled_params = []

        for template in self._parameters:
            if template.is_filled_by(fillers):
                filled_params.append(template.fill_slots(fillers))
            else:
                return ValueFactory.none()

        return self._function(*filled_params)

    def __hash__(self):
        return abs(hash(self._function) + hash(tuple(self._parameters)))  # list->tuple type casting by jmlee (2018.05.28)

    def __str__(self):
        result = self._function.__name__ + '('
        result += ','.join([str(param) for param in self._parameters])
        result += ')'

        return result

    def __eq__(self, other):
        if not isinstance(other, FunctionalTemplate):
            return False

        return self.__str__() == str(other)

    @dispatch()
    def get_function(self):
        return self._function
