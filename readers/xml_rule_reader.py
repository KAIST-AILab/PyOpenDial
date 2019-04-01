import random

import regex as re

from bn.values.value_factory import ValueFactory
from datastructs.math_expression import MathExpression
from domains.rules.conditions.basic_condition import BasicCondition, Relation
from domains.rules.conditions.complex_condition import ComplexCondition, BinaryOperator
from domains.rules.conditions.negated_condition import NegatedCondition
from domains.rules.conditions.void_condition import VoidCondition
from domains.rules.effects.basic_effect import BasicEffect
from domains.rules.effects.effect import Effect
from domains.rules.effects.template_effect import TemplateEffect
from domains.rules.parameters.complex_parameter import ComplexParameter
from domains.rules.parameters.fixed_parameter import FixedParameter
from domains.rules.parameters.single_parameter import SingleParameter
from domains.rules.rule import Rule, RuleType, RuleOutput
from templates.template import Template
from utils.xml_utils import XMLUtils


class XMLRuleReader:
    _id_counter = 1

    @staticmethod
    def get_rule(node):
        """
        Extracts the rule corresponding to the XML specification.

        :param node: topNode the XML node
        :return: the corresponding rule
        """

        # extracting the rule type
        rule_type = RuleType.PROB
        if 'util=' in XMLUtils.serialize(node):
            rule_type = RuleType.UTIL

        # setting the rule identifier
        try:
            rule_id = node.attrib['id']
        except:
            rule_id = 'rule' + str(XMLRuleReader._id_counter)
            XMLRuleReader._id_counter += 1

        # creating the rule
        rule = Rule(rule_id, rule_type)

        priority = 1
        if 'priority' in node.keys():
            priority = int(node.attrib['priority'])

        # extracting the rule cases
        for child_node in node:
            if child_node.tag == 'case':
                condition = XMLRuleReader._get_condition(child_node)
                output = XMLRuleReader._get_output(child_node, rule_type, priority)
                rule.add_case(condition, output)
            elif XMLUtils.has_content(node):
                if node.tag == '#text':
                    raise ValueError()

        return rule

    @staticmethod
    def _get_condition(case_node):
        """
        Returns the condition associated with the rule specification.

        :param case_node: the XML node
        :return: the associated rule condition
        """
        for child_node in case_node:
            if child_node.tag == 'condition':
                return XMLRuleReader._get_full_condition(child_node)

        return VoidCondition()

    @staticmethod
    def _get_output(case_node, rule_type, priority):
        """
        Returns the output associated with the rule specification.

        :param case_node: the XML node
        :param rule_type: the rule type
        :param priority: the rule priority
        :return: the output associated with the rule specification.
        """
        output = RuleOutput(rule_type)

        for child_node in case_node:
            # extracting an effect
            if child_node.tag == 'effect':
                effect = XMLRuleReader._get_full_effect(child_node, priority)
                if effect is None:
                    continue

                prob = XMLRuleReader._get_parameter(child_node, rule_type)
                output.add_effect(effect, prob)

        return output

    @staticmethod
    def _get_full_condition(condition_node):
        """
        Extracting the condition associated with an XML specification.

        :param condition_node: the XML node
        :return: the associated condition
        """
        sub_conditions = list()

        for child_node in condition_node:
            if XMLUtils.has_content(child_node):
                sub_condition = XMLRuleReader._get_sub_condition(child_node)
                sub_conditions.append(sub_condition)

        if len(sub_conditions) == 0:
            return VoidCondition()

        operator_str = None
        if 'operator' in condition_node.keys():
            operator_str = condition_node.attrib['operator'].lower().strip()

        if operator_str is not None:
            if operator_str == 'and':
                return ComplexCondition(sub_conditions, BinaryOperator.AND)
            elif operator_str == 'or':
                return ComplexCondition(sub_conditions, BinaryOperator.OR)
            elif operator_str == 'neg' or operator_str == 'not':
                negated_condition = sub_conditions[0] if len(sub_conditions) == 1 else ComplexCondition(sub_conditions, BinaryOperator.AND)
                return NegatedCondition(negated_condition)
            else:
                raise ValueError()

        return sub_conditions[0] if len(sub_conditions) == 1 else ComplexCondition(sub_conditions, BinaryOperator.AND)

    @staticmethod
    def _get_sub_condition(node):
        """
        Extracting a partial condition from a rule specification

        :param node: the XML node
        :return: the corresponding condition
        """
        # extracting a basic condition
        if node.tag == 'if':
            if 'var' not in node.keys():
                raise ValueError()

            variable_name = node.attrib['var']
            template = Template.create(variable_name)

            if template.is_under_specified():
                template = Template.create(str(template).replace('*', '{' + str(random.randint(1, 99)) + '}'))

            value_str = None
            if 'value' in node.keys():
                value_str = node.attrib['value']

            if value_str is not None:
                relation = XMLRuleReader._get_relation(node)
                condition = BasicCondition(variable_name, value_str, relation)
            else:
                if 'var2' not in node.keys():
                    raise ValueError()

                second_variable = node.attrib['var2']
                relation = XMLRuleReader._get_relation(node)
                condition = BasicCondition(variable_name, '{' + second_variable + '}', relation)

            for attrib_key in node.attrib.keys():
                if attrib_key not in ['var', 'var2', 'value', 'relation']:
                    raise ValueError()

            return condition

        # extracting a conjunction or disjunction
        if node.tag == 'or' or node.tag == 'and':
            conditions = list()

            for child_node in node:
                if XMLUtils.has_content(child_node):
                    conditions.append(XMLRuleReader._get_sub_condition(child_node))

            return ComplexCondition(conditions, BinaryOperator.OR if node.tag == 'or' else BinaryOperator.AND)

        # extracting a negated conjunction
        if node.tag == 'neg' or node.tag == 'not':
            conditions = list()

            for child_node in node:
                if XMLUtils.has_content(child_node):
                    conditions.append(XMLRuleReader._get_sub_condition(child_node))

            return conditions[0] if len(conditions) == 1 else ComplexCondition(conditions, BinaryOperator.AND)

        if XMLUtils.has_content(node):
            raise ValueError()

        return VoidCondition()

    @staticmethod
    def _get_relation(node):
        """
        Extracts the relation specified in a condition

        :param node: the XML node containing the relation
        :return: the corresponding relation
        """
        if 'relation' not in node.keys():
            return Relation.EQUAL

        relation = node.attrib['relation'].lower().strip()
        if relation == '=' or relation == 'equal':
            return Relation.EQUAL

        if relation == '!=' or relation == 'unequal':
            return Relation.UNEQUAL

        if relation == 'contains':
            return Relation.CONTAINS

        if relation == 'in':
            return Relation.IN

        if relation == 'length':
            return Relation.LENGTH

        if relation == '!contains':
            return Relation.NOT_CONTAINS

        if relation == '!in':
            return Relation.NOT_IN

        if relation == '>':
            return Relation.GREATER_THAN

        if relation == '<':
            return Relation.LOWER_THAN

        raise ValueError()

    @staticmethod
    def _get_full_effect(effect_node, priority):
        """
        Extracts a full effect from the XML specification.

        :param effect_node: the XML node
        :param priority: the rule priority
        :return: the corresponding effect
        """
        effects = list()

        for child_node in effect_node:
            if XMLUtils.has_content(child_node) and len(child_node.attrib) > 0:
                sub_effect = XMLRuleReader._get_sub_effect(child_node, priority)
                effects.append(sub_effect)

        return Effect(effects)

    @staticmethod
    def _get_sub_effect(node, priority):
        """
        Extracts a basic effect from the XML specification.

        :param node: the XML node
        :param priority: the rule priority
        :return: the corresponding basic effect
        """
        node_keys = node.keys()
        if 'var' not in node_keys:
            raise ValueError()

        variable_name = node.attrib['var']

        if 'value' in node_keys:
            value = node.attrib['value']
        elif 'var2' in node_keys:
            value = '{' + node.attrib['var2'] + '}'
        else:
            value = 'None'
        value = re.sub(r'\s+', ' ', value)

        exclusive = True
        if 'exclusive' in node_keys:
            if node.attrib['exclusive'].lower() == 'false':
                exclusive = False

        negated = 'relation' in node_keys and XMLRuleReader._get_relation(node) == Relation.UNEQUAL

        #  "clear" effect is outdated
        if node.tag.lower() == 'clear':
            value = 'None'

        # checking for other attributes
        for attrib_key in node_keys:
            if attrib_key not in ['var', 'var2', 'value', 'relation', 'exclusive']:
                raise ValueError()

        template_variable = Template.create(variable_name)
        template_value = Template.create(value)

        if template_variable.is_under_specified() or template_value.is_under_specified():
            return TemplateEffect(template_variable, template_value, priority, exclusive, negated)
        else:
            return BasicEffect(variable_name, ValueFactory.create(str(template_value)), priority, exclusive, negated)

    @staticmethod
    def _get_parameter(node, rule_type):
        """
        Returns the parameter described by the XML specification.

        :param node: the XML node
        :param rule_type: the rule type
        :return: the parameter representation
        """
        if rule_type is RuleType.PROB:
            if 'prob' in node.keys():
                return XMLRuleReader._get_inner_parameter(node.attrib['prob'])
            else:
                return FixedParameter(1.)

        if rule_type is RuleType.UTIL:
            if 'util' in node.keys():
                return XMLRuleReader._get_inner_parameter(node.attrib['util'])

        raise ValueError()

    @staticmethod
    def _get_inner_parameter(str_val):
        """
        Returns the parameter described by the XML specification.

        :param str_val: XML node string
        :return: the parameter representation
        """
        # we first try to extract a fixed value
        try:
            return FixedParameter(float(str_val))
        except ValueError:
            # if it fails, we extract an actual unknown parameter
            if '{' in str_val or '+' in str_val or '*' in str_val or '-' in str_val:
                # if we have a complex expression of parameters
                return ComplexParameter(MathExpression(str_val))
            else:
                # else, we extract a stochastic parameter
                searcher = re.compile('.+(\\[[0-9]+\\])').search(str_val)
                if searcher:
                    dimension = int(searcher.group(1).replace('[', '').replace(']', ''))
                    parameter_id = str_val.replace(searcher.group(1), '').strip()
                    return SingleParameter(parameter_id, dimension)
                else:
                    return SingleParameter(str_val)
