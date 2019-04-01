import abc
import logging
from collections import OrderedDict

import regex as re
from multipledispatch import dispatch

from bn.values.value import Value

dispatch_namespace = dict()


class Graph:
    """
    Representation of a relational structure as a directed acyclic graph. The two
    parameters V and R respectively express the types of values associated with each
    node and each relation. In addition to its main content, each node can also
    express a set of attributes (such as POS tags, named entities, timing information,
    etc.).

    The string representation of the graph is similar to the one used in the Stanford
    NLP package. For instance, the string [loves subject>John object>Mary] represents
    a graph with three nodes (loves, John and Mary), with a relation labelled
    "subject" from "loves" to "John", and a relation labelled "object" between "loves"
    and "Mary". Brackets are used to construct embedded graphs, such as for instance
    [eats subject>Pierre object>[apple attribute>red]], which is a graph with four
    nodes, where the node "apple" is itself the governor of the node "red".

    The optional attributes are indicated via a | bar followed by a key:value pair
    right after the node content. For instance, "loves|pos:VB" indicates that the pos
    attribute for the node has the value "VB". To incorporate several attributes, you
    can simply add additional | bars, like this: loves|pos:VB|index:1.

    Finally, it is also possible to construct graph with more than one root by
    including several brackets at the top level, such as [eats subject>Pierre][drinks
    subject>Milen].

    The class is abstract, and its extension requires the definition of three methods
    that define how the values V and R can be created from string, and how values V
    can be copied.
    """
    __metaclass__ = abc.ABCMeta

    # logger
    log = logging.getLogger('PyOpenDial')

    _value_regex = r'([^\[\]\s\|]+)((?:\|\w+:[^\[\]\s\|]+)*)'
    _value_pattern = re.compile(_value_regex)
    _graph_pattern = re.compile(r"\[" + _value_regex + r"((\s+\S+>" + _value_regex + r")*)\]")

    def __init__(self, arg1=None):
        if arg1 == None:
            """
            Constructs an empty graph
            """
            self._roots = []
            self._nodes = []
            self._str = ''

        elif isinstance(arg1, str):
            str_representation = arg1
            """
            Constructs a graph from a string
    
            :param str_representation: the string representation for the graph
            """
            self._roots = []
            self._nodes = []

            str_representation = str_representation.strip().replace('> ', '>')
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


    def __hash__(self):
        """
        Returns the hashcode for the graph
        """
        return self._str.__hash__()

    def __eq__(self, other):
        """
        Returns true if the object is a graph with the same content.
        """
        if not isinstance(other, Graph):
            return False

        return self._str == str(other)

    def __str__(self):
        """
        Returns the string representation for the graph
        """
        return self._str

    @staticmethod
    @dispatch(str, namespace=dispatch_namespace)
    def is_relational(str_representation):
        """
        Returns true if the string represents a relational structure, else false.

        :param str_representation: the string to check
        :return: true if the string encodes a graph, else false
        """
        if not str_representation.startswith('['):
            return False

        if not str_representation.endswith(']'):
            return False

        if '>' not in str_representation:
            return False

        return Graph._graph_pattern.search(str_representation) is not None

    @dispatch(str)
    @abc.abstractmethod
    def create_value(self, str_representation):
        """
        Creates a value of type V from a string

        :param str_representation: the string
        :return: the corresponding value
        """
        raise NotImplementedError()

    @dispatch(str)
    @abc.abstractmethod
    def create_relation(self, str_representation):
        """
        Creates a value of type R from a string

        :param str_representation: the string
        :return: the corresponding value
        """
        raise NotImplementedError()

    @dispatch(Value)
    @abc.abstractmethod
    def copy_value(self, value):
        """
        Copies the value
        :param value: the value to copy
        :return: the copied value
        """
        raise NotImplementedError()

    @dispatch()
    def get_nodes(self):
        """
        Returns the nodes of the graph

        :return: the nodes
        """
        return self._nodes

    @dispatch()
    def get_roots(self):
        """
        Returns the roots of the graph

        :return: the roots
        """
        return self._roots

    @dispatch(str)
    def _create_node(self, str_representation):
        """
        Creates a new node from the string representation. This method is called
        recursively to build the full graph structure.

        :param str_representation: the string
        :return: the corresponding node
        """
        searcher = Graph._graph_pattern.search(str_representation)

        while searcher is not None:
            if searcher.start() > 0 or searcher.end() < len(str_representation):
                node = self._create_node(searcher.group(0))
                str_representation = str_representation[:searcher.start()] + str(id(node)) + str_representation[searcher.end():]
                searcher = Graph._graph_pattern.search(str_representation)
            else:
                content = searcher.group(1)
                node = Node(self.create_value(content))

                attributes = searcher.group(2)
                attributes = attributes[1:] if len(attributes) > 0 else attributes
                for attribute in attributes.split('|'):
                    if len(attribute) == 0:
                        continue
                    attribute = attribute.split(':')
                    node.add_attributes(attribute[0], self.create_value(attribute[1]))

                relations = searcher.group(3).split(' ')
                for relation in relations:
                    if len(relation) == 0 or '>' not in relation:
                        continue

                    relation = relation.split('>')
                    relation_key = self.create_relation(relation[0])
                    relation_content = relation[1]

                    child_node = None
                    for _node in self._nodes:
                        if str(id(_node)) == relation_content:
                            child_node = _node
                            break

                    node.add_child(relation_key, self._create_node("[%s]" % relation_content) if child_node is None else child_node)
                self._nodes.insert(0, node)
                return node
        raise ValueError()


class NodeWrapper:
    pass


class Node(NodeWrapper):
    """
    Representation of an individual node, with a content, optional attributes and
    outgoing relations.
    """

    def __init__(self, arg1):
        if isinstance(arg1, object):
            content = arg1
            """
            Creates a new node with the given content
    
            :param content: the content
            """
            self._content = content
            self._children = OrderedDict()
            self._attributes = dict()

        else:
            raise NotImplementedError()

    def __str__(self):
        """
        Returns a string representation of the node and its descendants
        """
        result = str(self._content) + ''.join(['|%s:%s' % (key, str(value)) for key, value in self._attributes.items()])
        if len(self._children) == 0:
            return result

        return '[' + result + ''.join([' %s>%s' % (key, str(value)) for key, value in self._children.items()]) + ']'

    def __eq__(self, other):
        """
        Returns true if the object is a graph with the same content.
        """
        if not isinstance(other, Node):
            return False

        if self._content != other._content:
            return False

        if self._children != other._children:
            return False

        if self._attributes != other._attributes:
            return False

        return True

    def __hash__(self):
        """
        Returns the hashcode for the node
        """
        return hash(self._content) - hash(frozenset(self._children.items())) + hash(frozenset(self._attributes.items()))

    @dispatch(str, object)
    def add_attributes(self, key, value):
        """
        Adds a new attribute to the node

        :param key: the attribute label
        :param value: the corresponding value
        """
        self._attributes[key] = value

    @dispatch(object, NodeWrapper)
    def add_child(self, relation, node):
        """
        Adds a new outgoing relation to the node. Throws an exception if a cycle
        is found.

        :param relation: the relation label
        :param node: the dependent node
        """
        if self in node.get_descendants():
            raise ValueError()

        self._children[relation] = node
        self._children = OrderedDict(sorted(self._children.items(), key=lambda t: t[0]))

    @dispatch()
    def get_content(self):
        """
        Returns the node content
        """
        return self._content

    @dispatch()
    def get_relations(self):
        """
        returns the relation labels going out of the node
        """
        return sorted(set(self._children.keys()))

    @dispatch(object)
    def get_child(self, relation):
        """
        Returns the node that is a child of the current node through the given
        relation. Returns null if the child cannot be found.

        :param relation: the labelled relation
        :return: the corresponding child node
        """
        return self._children[relation]

    @dispatch()
    def get_attributes(self):
        """
        Returns the set of attribute keys.
        :return: the keys
        """
        return set(self._attributes.keys())

    @dispatch(str)
    def get_attr_value(self, key):
        """
        Returns the attribute value for the given key, if it exists. Else returns null.
        """
        return self._attributes[key]

    @dispatch()
    def get_children(self):
        """
        Returns the set of children nodes

        :return: the children nodes
        """
        return list(self._children.values())

    @dispatch()
    def get_descendants(self):
        """
        Returns the set of all descendant nodes.

        :return: the descendant nodes
        """
        descendants = set()
        to_process = list(self._children.values())

        while len(to_process) > 0:
            node = to_process.pop(0)
            descendants.add(node)
            to_process.extend(node.get_children())

        return descendants

    @dispatch()
    def copy(self):
        """
        Copies the node

        :return: the copy
        """
        node = Node(self._copy_value_func(self._content), self._copy_value_func)
        for attr_key in self._attributes.keys():
            node.add_attributes(attr_key, self._copy_value_func(self._attributes[attr_key]))

        for child_key in self._children.keys():
            node.add_child(child_key, self._children[child_key].copy())

        return node

    @dispatch(NodeWrapper)
    def merge(self, other):
        """
        Merges the node with another one (if two values are incompatible, the
        content of the other node takes precedence).

        :param other: otherGraphNode the other node
        :return: the merged node
        """
        merged_node = other.copy()
        for attr_key in self._attributes:
            merged_node.add_attribute(attr_key, self._copy_value_func(self._attributes[attr_key]))

        for child_key in self._children.keys():
            if child_key in merged_node._children:
                merged_child_node = merged_node._children[child_key]
                merged_child_node = self._children[child_key].merge(merged_child_node)
                merged_node.add_child(child_key, merged_child_node)
            else:
                merged_node.add_child(child_key, self._children[child_key])

        return merged_node
