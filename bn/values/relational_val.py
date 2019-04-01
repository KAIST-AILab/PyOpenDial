from copy import copy

from bn.values.value import Value
from datastructs.graph import Graph

import logging
from multipledispatch import dispatch


class RelationalVal(Graph, Value):
    """
    Representation of a relational value. Its extends a graph where the nodes contains
    Value objects, and the relation are plain strings. See the Graph class for more
    information about the syntax for the graph.
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    def __init__(self, arg1=None):
        if arg1 is None:
            """
            Creates an empty relational structure.
            """
            # TODO: change to: super(Graph, self).__init__()
            # TODO: dispatch cannot find super class
            self._roots = []
            self._nodes = []
            self._str = ''
        elif isinstance(arg1, str):
            value = arg1
            """
            Creates a relational structure from a string representation.

            :param value: string
            """
            # TODO: change to: super(Graph, self).__init__(value)
            # TODO: dispatch cannot find super class
            self._roots = []
            self._nodes = []

            str_representation = value.strip().replace('> ', '>')
            bracket_cnt = 0

            tmp = []
            for c in str_representation:
                tmp.append(c)

                if c == '[':
                    bracket_cnt += 1
                elif c == ']':
                    bracket_cnt -= 1

                    if bracket_cnt == 0:
                        node = self._create_node(''.join(tmp))
                        self._roots.append(node)

            self._str = ''.join([str(root) for root in self._roots])
        else:
            raise NotImplementedError()

    def __lt__(self, other):
        """
        Compares the value with another one (based on their string).

        :param other: the object to compare
        :return: boolean result of comparison
        """
        return str(self) < str(other)

    def __len__(self):
        """
        Returns the number of nodes in the graph.

        :return: the number of nodes
        """
        return len(self.get_nodes())

    def __copy__(self):
        """
        Copies the relational structure.

        :return: the copy
        """
        return RelationalVal(str(self))

    def __contains__(self, item):
        """
        Returns true if the value is contained in the relation structure.

        :param item: the value
        :return: true if the item is same value
        """
        return str(item) in str(self)

    @dispatch()
    def get_sub_values(self):
        """
        Returns the collection of values in the relational structure.

        :return: the list of values
        """
        return [node.get_content() for node in self.get_nodes()]

    @dispatch(Value)
    def concatenate(self, value):
        """
        Concatenates two relational structures (by juxtaposing their roots).

        :param value: the value to concatenate
        :return: the concatenated result
        """
        if not isinstance(value, RelationalVal):
            raise ValueError()

        return RelationalVal(str(self) + str(value))

    @dispatch()
    def is_empty(self):
        """
        Returns true if the structure does not contain any nodes, else false.

        :return: true if the structure does not contain any nodes, else false
        """
        return len(self.get_nodes()) == 0

    @dispatch(str)
    def create_value(self, str_representation):
        """
        Creates a value from a string representation within the graph.

        :param str_representation: string representation
        :return: the value
        """
        from bn.values.value_factory import ValueFactory
        return ValueFactory.create(str_representation)

    @dispatch(str)
    def create_relation(self, str_representation):
        """
        Creates a relation from a string representation within the graph.

        :param str_representation: string representation
        :return: the relation
        """
        return str_representation

    @dispatch(Value)
    def copy_value(self, value):
        """
        Copies a value.

        :param value: the value
        :return: the copy
        """
        return copy(value)
