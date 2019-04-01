from collections import Collection

import regex as re
import numpy as np
from bn.values.array_val import ArrayVal
from bn.values.boolean_val import BooleanVal
from bn.values.double_val import DoubleVal
from bn.values.none_val import NoneVal
from bn.values.relational_val import RelationalVal
from bn.values.set_val import SetVal
from bn.values.string_val import StringVal
from bn.values.custom_val import CustomVal
from bn.values.value import Value
from datastructs.graph import Graph
from utils.py_utils import get_class, Singleton
import logging
from multipledispatch import dispatch
from settings import Settings

dispatch_namespace = dict()

class ValueFactory:
    """
    Factory for creating variable values.
    """
    _none_value = NoneVal()
    _double_pattern = re.compile(r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$')
    _array_pattern = re.compile(r'\[([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?,\s*)*([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\]')
    _set_pattern = re.compile(r'[/\w\-_\.\^\=\s]*([\[\(][/\w\-_,\.\^\=\s\(]+\)*[\]\)])?')
    _custom_class_pattern = re.compile(r'^@[^\(\)]*$')
    _custom_function_pattern = re.compile(r'^@[^\(\)]+\(.*\)$')

    # logger
    log = logging.getLogger('PyOpenDial')

    @staticmethod
    @dispatch(str, namespace=dispatch_namespace)
    def create(value):
        """
        Creates a new value based on the provided string representation. If the string
        contains a numeric value, "true", "false", "None", or opening and closing
        brackets, convert it to the appropriate values. Else, returns a string value.

        :param value: the string representation for the value
        :return: the resulting value
        """
        if value == None:
            return NoneVal()

        if ValueFactory._double_pattern.search(value):
            return DoubleVal(float(value))
        elif value.lower() == 'true':
            return BooleanVal(True)
        elif value.lower() == 'false':
            return BooleanVal(False)
        elif value.lower() == 'none':
            return ValueFactory._none_value
        elif ValueFactory._array_pattern.match(value):
            value_list = list()
            value_str_list = value[1:-1].split(',')
            for value_str_item in value_str_list:
                value_list.append(float(value_str_item))

            return ArrayVal(np.array(value_list))
        elif value.startswith('[') and value.endswith(']'):
            if Graph.is_relational(value):
                relation_value = RelationalVal(value)
                if not relation_value.is_empty():
                    return relation_value

            sub_values = list()
            for match in ValueFactory._set_pattern.finditer(value[1:-1]):
                sub_value = match.group(0).strip()
                if len(sub_value) > 0:
                    sub_values.append(ValueFactory.create(sub_value))

            return SetVal(sub_values)
        elif ValueFactory._custom_class_pattern.match(value):
            class_name = value[1:]
            custom_value = get_class(class_name)()
            if isinstance(custom_value, Singleton):
                return CustomVal(custom_value)
            else:
                raise ValueError("Custom class should inherit utils.py_utils.Singleton")
        elif ValueFactory._custom_function_pattern.match(value):
            function_name = value.split("(")[0][1:]
            params = value.split("(")[1][:-1].split(",")
            params.remove('')
            if function_name in Settings._functions:
                func = Settings._functions[function_name]
                func_result = func(*params)
                if isinstance(func_result, float):
                    return DoubleVal(func_result)
                elif isinstance(func_result, bool):
                    return BooleanVal(func_result)
                elif func_result is None:
                    return ValueFactory._none_value
                elif isinstance(func_result, np.ndarray):
                    return ArrayVal(func_result)
                elif isinstance(func_result, set):
                    return SetVal(func_result)
                elif isinstance(func_result, str):
                    return StringVal(func_result)
                else:
                    raise ValueError("Not supported return type %s" % type(func_result))
            else:
                raise ValueError("Function %s is not defined." % function_name)

        else:
            return StringVal(value)

    @staticmethod
    @dispatch(float, namespace=dispatch_namespace)
    def create(value):
        """
        Returns a double value given the double

        :param value: the float
        :return: the value
        """
        return DoubleVal(value)

    @staticmethod
    @dispatch(bool, namespace=dispatch_namespace)
    def create(value):
        """
        Returns the boolean value given the boolean

        :param value: the boolean
        :return: the boolean value
        """
        return BooleanVal(value)

    @staticmethod
    @dispatch((list, Collection), namespace=dispatch_namespace)
    def create(values):
        """
        Returns the set value given the values

        :param values: the values
        :return: the set value
        """
        if len(values) == 0 or isinstance(next(iter(values)), Value):
            return SetVal(values)
        if isinstance(values[0], float):
            return ArrayVal(np.array(values))

    @staticmethod
    @dispatch(namespace=dispatch_namespace)
    def none():
        """
        Returns the none value.

        :return: the none value
        """
        return ValueFactory._none_value

    @staticmethod
    @dispatch(Value, Value, namespace=dispatch_namespace)
    def concatenate(v1, v2):
        """
        Returns the concatenation of the two values.

        :param v1: the value
        :param v2: the value
        :return: the concatenation of the two values
        """
        if isinstance(v1, StringVal) and isinstance(v2, StringVal):
            return str(v1) + ' ' + str(v2)
        elif isinstance(v1, NoneVal):
            return v2
        elif isinstance(v2, NoneVal):
            return v1
        else:
            ValueFactory.log.warning("concatenation not implemented for %s + %s" % (v1, v2))
            return NoneVal()
