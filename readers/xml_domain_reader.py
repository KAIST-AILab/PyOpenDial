import os
from pathlib import Path

import importlib
import regex as re

from dialogue_state import DialogueState
from domains.domain import Domain
from domains.model import Model
from readers.xml_rule_reader import XMLRuleReader
from readers.xml_state_reader import XMLStateReader
from utils.xml_utils import XMLUtils
from bn.nodes.custom_utility_function import CustomUtilityFunction

"""
XML reader for dialogue domains.

@author Pierre Lison (plison@ifi.uio.no)
"""
class XMLDomainReader:
    @staticmethod
    def extract_empty_domain(file_name):
        """
        Extract a empty domain from the XML domain specification, only setting the
        source file and its possible imports. This method is used to be able to
        extract the source and import files in case the domain is ill-formed. You can
        usually safely ignore this method.

        :param file_name: the filename of the top XML file
        :return: the extracted dialogue domain
        """
        XMLDomainReader.extract_domain(file_name, False)

    @staticmethod
    def extract_domain(top_domain_file, full_extract=True):
        """
        Extract a dialogue domain from the XML specification

        :param top_domain_file: the filename of the top XML file
        :param full_extract: whether to extract the full domain or only the files
        :return: the extracted dialogue domain
        """
        # create a new, empty domain
        domain = Domain()

        # determine the root path and filename
        fl = open(top_domain_file, 'r')
        domain.set_source_file(Path(top_domain_file))

        # extract the XML document
        document = XMLUtils.get_xml_document(fl)
        main_node = XMLUtils.get_main_node(document)

        root_path = Path(top_domain_file).parent

        for child in main_node:
            domain = XMLDomainReader.extract_partial_domain(child, domain, root_path, full_extract)

        return domain

    @staticmethod
    def extract_partial_domain(main_node, domain, root_path, full_extract):
        """
        Extracts a partially specified domain from the XML node and add its content to
        the dialogue domain.

        :param main_node: main XML node
        :param domain: dialogue domain
        :param root_path: root path (necessary to handle references)
        :param full_extract: whether to extract the full domain or only the files
        :return: the augmented dialogue domain
        """
        tag = main_node.tag

        if tag == 'domain':
            # extracting rule-based probabilistic model
            for child in main_node:
                domain = XMLDomainReader.extract_partial_domain(child, domain, root_path, full_extract)
        elif tag == 'import':
            # extracting imported references
            try:
                file_name = main_node.attrib['href']
                file_path = str(root_path) + os.sep + file_name
                fl = Path(file_path)
                domain.add_imported_files(fl)
                sub_document = XMLUtils.get_xml_document(file_path)
                domain = XMLDomainReader.extract_partial_domain(XMLUtils.get_main_node(sub_document), domain, root_path, full_extract)
            except:
                raise ValueError()

        if not full_extract:
            return domain

        if tag == 'settings':
            # extracting settings
            settings = XMLUtils.extract_mapping(main_node)
            domain.get_settings().fill_settings(settings)
        if tag == 'function':
            # extracting custom functions
            # try:
            domain_function_name = main_node.attrib['name'].strip()

            module_name, actual_function_name = main_node.text.rsplit('.', 1)
            mod = importlib.import_module(module_name)
            func = getattr(mod, actual_function_name)

            domain.get_settings().add_function(domain_function_name, func)
            # except:
            #     raise ValueError()
        if tag == 'initialstate':
            # extracting initial state
            state = XMLStateReader.get_bayesian_network(main_node)
            domain.set_initial_state(DialogueState(state))
        if tag == 'model':
            # extracting rule-based probabilistic model
            model = XMLDomainReader._create_model(main_node)
            domain.add_model(model)
        if tag == 'parameters':
            # extracting parameters
            parameters = XMLStateReader.get_bayesian_network(main_node)
            domain.set_parameters(parameters)
        if XMLUtils.has_content(main_node):
            if main_node == '#text':  # TODO: main_node -> main_node.tag ??
                raise ValueError()

        return domain

    @staticmethod
    def _create_model(node):
        """
        Given an XML node, extracts the rule-based model that corresponds to it.

        :param node: the XML node
        :return: the corresponding model
        """
        model = Model()
        for child in node:
            if child.tag == 'rule':
                rule = XMLRuleReader.get_rule(child)
                model.add_rule(rule)
            elif child.tag == 'custom-utility':
                action_node_id = child.attrib['var']
                simulation_action_node_id = child.attrib['simvar']
                custom_utility_function_name = child.attrib['function'].strip()
                module_name, actual_function_name = custom_utility_function_name.rsplit('.', 1)
                mod = importlib.import_module(module_name)
                func = getattr(mod, actual_function_name)

                custom_utility_function = CustomUtilityFunction(action_node_id, simulation_action_node_id, func)
                model.set_custom_utility_function(custom_utility_function)
            elif XMLUtils.has_content(child):
                if child.tag == '#text':
                    raise ValueError()

        if 'trigger' in node.keys():
            trigger_str = node.attrib['trigger']
            matcher_list = re.compile(r'([\w\*\^_\-\[\]\{\}]+(?:\([\w\*,\s\^_\-\[\]\{\}]+\))?)[\w\*\^_\-\[\]\{\}]*').findall(trigger_str)
            for matcher in matcher_list:
                model.add_trigger(matcher)

        if 'blocking' in node.keys():
            if node.attrib['blocking'].lower() == 'true':
                blocking = True
            elif node.attrib['blocking'].lower() == 'false':
                blocking = False
            else:
                raise ValueError()
            model.set_blocking(blocking)

        if 'id' in node.keys():
            model.set_id(node.attrib['id'])

        return model
