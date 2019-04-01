import xml.etree.ElementTree as ET
from xml.dom import minidom
from xml.etree.ElementTree import ElementTree, Element
from io import IOBase
import logging

from multipledispatch import dispatch

from dialogue_state import DialogueState

dispatch_namespace = dict()


class XMLUtils:
    """
    Utility functions for manipulating XML content
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    @staticmethod
    @dispatch(str, namespace=dispatch_namespace)
    def get_xml_document(filename):
        return ET.parse(filename)  # ET.fromstring(value)

    @staticmethod
    @dispatch(IOBase, namespace=dispatch_namespace)
    def get_xml_document(source):
        return ET.parse(source)

    @staticmethod
    @dispatch(Element, namespace=dispatch_namespace)
    def serialize(node):
        """
        Serialises the XML node into a string.

        :param node: the XML node
        :return: the corresponding string
        """
        return str(ET.tostring(node, encoding='utf8', method='xml', short_empty_elements=True), 'utf-8')

    @staticmethod
    @dispatch(str, namespace=dispatch_namespace)
    def new_xml_document(root_element_name):
        """
        Create a new XML, empty document

        :return: the empty XML document
        """
        try:
            root = Element(root_element_name)
            return ElementTree(root)
        except Exception as e:
            XMLUtils.log.warning(str(e))
            raise ValueError("cannot create XML file")

    @staticmethod
    @dispatch(ElementTree, str, namespace=dispatch_namespace)
    def write_xml_document(doc, filename):
        """
        Writes the XML document to the particular file specified as argument

        :param doc: the document
        :param filename: the path to the file in which to write the XML data writing operation fails
        """
        xmlstr = minidom.parseString(ET.tostring(doc.getroot(), encoding='utf-8')).toprettyxml(indent="   ")
        with open(filename, "w") as f:
            f.write(xmlstr)

    @staticmethod
    @dispatch(ElementTree, namespace=dispatch_namespace)
    def write_xml_string(doc):
        raise NotImplementedError()

    @staticmethod
    @dispatch(str, namespace=dispatch_namespace)
    def load_xml_from_string(xml):
        raise NotImplementedError()

    @staticmethod
    @dispatch(ElementTree, namespace=dispatch_namespace)
    def get_main_node(document):
        """
        Returns the main node of the XML document

        :param document: the XML document
        :return: the main node
        """
        # TODO: 이 함수 좀 이상함..
        root = document.getroot()
        if root.tag != '#text' and root.tag != '#comment':
            return root

        raise ValueError("main node in XML file could not be retrieved")

    @staticmethod
    @dispatch(str, namespace=dispatch_namespace)
    def extract_mapping(settings_file):
        """
        Extract the settings from the XML file.

        :param settings_file: the file containing the settings
        :return: the resulting list of properties
        """
        doc = XMLUtils.get_xml_document(settings_file)
        mapping = XMLUtils.get_main_node(doc)
        return mapping

    @staticmethod
    @dispatch(Element, namespace=dispatch_namespace)
    def extract_mapping(main_node):
        """
        Extract the settings from the XML node.

        :param main_node: the XML node containing the settings
        :return: the resulting list of properties
        """
        settings = dict()

        for child_node in main_node:
            if child_node.tag != '#text' and child_node.tag != '#comment':
                key_val = child_node.tag.strip()
                settings[key_val] = child_node.text

        return settings

    @staticmethod
    @dispatch(str, str, namespace=dispatch_namespace)
    def validate_xml(dial_specs, schema_file):
        raise NotImplementedError()

    @staticmethod
    @dispatch(ElementTree, namespace=dispatch_namespace)
    def extract_included_files(xml_document):
        raise NotImplementedError()

    @staticmethod
    @dispatch(object, str, str, namespace=dispatch_namespace)  # object: DialogueSystem
    def import_content(system, file, tag):
        """
        Imports a dialogue state or prior parameter distributions.

        :param system: the dialogue system
        :param file: the file that contains the state or parameter content
        :param tag: the expected top XML tag. into the system
        """
        from readers.xml_state_reader import XMLStateReader
        if tag == "parameters":
            parameters = XMLStateReader.extract_bayesian_network(file, tag)
            for old_param in system.get_state().get_parameter_ids():
                if not parameters.has_chance_node(old_param):
                    parameters.add_node(system.get_state().get_chance_node(old_param))
            system.get_state().set_parameters(parameters)
        else:
            state = XMLStateReader.extract_bayesian_network(file, tag)
            system.add_content(DialogueState(state))

    @staticmethod
    @dispatch(object, str, str, namespace=dispatch_namespace)  # object: DialogueSystem
    def export_content(system, file, tag):
        """
        Exports a dialogue state or prior parameter distributions.

        :param system: the dialogue system
        :param file: the file in which to write the state or parameter content
        :param tag: the expected top XML tag. from the system
        """
        doc = XMLUtils.new_xml_document(tag)
        parameter_ids = system.get_state().get_parameter_ids()
        other_vars_ids = system.get_state().get_chance_node_ids()
        for parameter in parameter_ids:
            other_vars_ids.remove(parameter)

        variables = parameter_ids if tag == "parameters" else other_vars_ids
        param_xml = system.get_state().generate_xml(variables)
        param_xml.tag = tag

        doc._root = param_xml
        XMLUtils.write_xml_document(doc, file)

    @staticmethod
    @dispatch(Element, namespace=dispatch_namespace)
    def has_content(node):
        """
        Returns true if the node has some actual content (other than a comment or an
        empty text).

        :param node: the XML node
        :return: true if the node contains information, false otherwise
        """
        if node.tag == '#comment':
            return False

        if node.tag == '#text':
            return len(node.text.strip()) > 0

        return True
