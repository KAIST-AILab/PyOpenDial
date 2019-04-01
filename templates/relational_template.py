from multipledispatch import dispatch
from copy import copy

from bn.values.relational_val import RelationalVal
from bn.values.value_factory import ValueFactory
from datastructs.assignment import Assignment
from datastructs.graph import Graph, Node
from templates.regex_template import RegexTemplate
from templates.template import Template


class RelationalTemplate(Graph, Template):
    """
    Template for a relational structure. Both the node content and relations can be
    "templated" (i.e. underspecified). See the Graph class for more details on the
    string representation.
    """

    def __init__(self, str_representation):
        if not isinstance(str_representation, str):
            raise NotImplementedError("UNDEFINED PARAMETERS")
        """
        Creates a new template from a string representation

        :param str_representation: string
        """
        # TODO: change to: super(Graph, self).__init__(value)
        # TODO: dispatch cannot find super class
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

        self._slots = set()
        for node in self.get_nodes():
            self._slots.update(node.get_content().get_slots())
            for relation in node.get_relations():
                self._slots.update(relation.get_slots())

    @dispatch(str)
    def match(self, str_val):
        """
        Returns the result of a full match between the template and the string. (this
        is performed using an equivalent regex template).
        """
        return RegexTemplate(self.__str__()).match(str_val)

    @dispatch(str, int)
    def find(self, str_val, max_results):
        """
        Searches for the occurrences of the relational template in the string (if the
        string is itself a relational structure). Else returns an empty list.
        """
        val = ValueFactory.create(str_val)
        if isinstance(val, RelationalVal):
            return self.get_matches(val)

        return list()

    @dispatch(Assignment)
    def is_filled_by(self, input):
        """
        Returns true if all slots are filled by the assignment, else returns false.
        """
        return input.contains_vars(self._slots)

    @dispatch(Assignment)
    def fill_slots(self, fillers):
        """
        Fills the slots in the template and returns the result.
        """
        fillers.filter_values(lambda v: str(v) != "")
        return RegexTemplate(str(self)).fill_slots(fillers)

    @dispatch()
    def get_slots(self):
        """
        Returns the slots in the template
        """
        return self._slots

    @dispatch()
    def is_under_specified(self):
        """
        Returns true
        """
        return True

    @dispatch(RelationalVal)
    def get_matches(self, rel_val):
        """
        Returns the list of occurrences of the template in the relational structure.

        :param rel_val: the relational structure to search in
        :return: the corresponding matches for the template
        """
        results = list()
        for root in self.get_roots():
            for node in rel_val.get_nodes():
                results.extend(self._get_matches(root, node))

        return results

    @staticmethod
    @dispatch(Node, Node)
    def _get_matches(template_node, rel_val_node):
        """
        Returns the list of possible matches between the template node and a node
        inside a relational value.

        :param template_node: the node in the template
        :param rel_val_node: the node in the relational value
        :return: the list of possible matches (may be empty)
        """
        # first checks whether the contents are matching
        content_match = template_node.get_content().match(str(rel_val_node.get_content()))
        if not content_match.is_matching():
            return list()

        # checks whether the attributes are matching
        for attribute in template_node.get_attributes():
            if attribute not in rel_val_node.get_attributes():
                return list()

            template_attr_val = template_node.get_attr_value(attribute)
            rel_val_attr_val = rel_val_node.get_attr_value(attribute)

            attr_match = template_attr_val.match(str(rel_val_attr_val))
            if not attr_match.is_matching():
                return list()

            content_match.add_assignment(attr_match)

        # creates for each template relation a list of possible matches
        results = list()
        for relation in template_node.get_relations():
            relation_results = RelationalTemplate._get_matches(
                relation, template_node.get_child(relation), rel_val_node)
            if len(relation_results) == 0:
                return list()
            else:
                results.append(relation_results)

        # generates the combination of all matches
        results.append([content_match])
        return RelationalTemplate._flatten_results(results)

    @staticmethod
    @dispatch(Template, Node, Node)
    def _get_matches(relation, template_node, rel_val_node):
        """
        Searches for all children of vNode that satisfy the template relation rel and
        also match the template node tSubNode.

        :param relation: the template relation
        :param template_node: the template node
        :param rel_val_node: the node in the relational value
        :return: the list of corresponding matches
        """
        relation_results = list()

        if str(relation) == '+':
            for descendant in rel_val_node.get_descendants():
                sub_matches = RelationalTemplate._get_matches(template_node, descendant)
                relation_results.extend(sub_matches)

            return relation_results

        for rel_val_relation in rel_val_node.get_relations():
            relation_match = relation.match(rel_val_relation)
            if not relation_match.is_matching():
                continue

            rel_val_sub_node = rel_val_node.get_child(rel_val_relation)
            sub_matches = RelationalTemplate._get_matches(template_node, rel_val_sub_node)
            for sub_match in sub_matches:
                sub_match.add_assignment(relation_match)
                relation_results.append(sub_match)

        return relation_results

    @staticmethod
    @dispatch(list)
    def _flatten_results(input_str):
        """
        Creates the combination of all match results.

        :param input_str: the list of all results for each template relation
        :return: the "flattened" combination of all matches
        """
        results = list()

        results.extend(input_str.pop(0))
        for relation_results in input_str:
            new_results = list()
            for cur_results in results:
                for rel_result in relation_results:
                    new_result = copy(cur_results)
                    new_result.add_assignment(rel_result)
                    new_results.append(new_result)

            results = new_results
        return results

    @dispatch(str)
    def create_value(self, str_representation):
        """
        Creates a template from its string representation
        """
        return Template.create(str_representation)

    @dispatch(str)
    def create_relation(self, str_representation):
        """
        Creates a template from its string representation
        """
        return Template.create(str_representation)

    @dispatch(Template)
    def copy_value(self, template):
        """
        Copies a template
        """
        return Template.create(str(template))
