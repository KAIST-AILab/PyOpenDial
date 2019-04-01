from datastructs.assignment import Assignment
from dialogue_state import DialogueState
from readers.xml_state_reader import XMLStateReader
from utils.xml_utils import XMLUtils
import logging
from multipledispatch import dispatch
import xml.etree.ElementTree as ET


class XMLDialogueReader:
    """
    XML reader for previously recorded dialogues.
    """
    # logger
    log = logging.getLogger('PyOpenDial')

    @staticmethod
    @dispatch(str)
    def extract_dialogue(data_file):
        """
        Extracts the dialogue specified in the data file. The result is a list of
        dialogue state (one for each turn).

        :param data_file: the XML file containing the turns
        :return: the list of dialogue state
        """
        doc = XMLUtils.get_xml_document(data_file)
        main_node = XMLUtils.get_main_node(doc)

        f = open(data_file)
        root_path = f.name

        sample = []

        for node in main_node:
            node_name = node.keys()[0]
            if "Turn" in node_name:
                state = DialogueState(XMLStateReader.get_bayesian_network(node))
                sample.append(state)

                if node_name == "systemTurn" and state.has_chance_node("a_m"):
                    assign = Assignment("a_m", state.query_prob("a_m").get_best())
                    state.add_evidence(assign)

            elif node_name == "wiazard":
                assign = Assignment.create_from_string(node.get_first_child().get_node_value().trim())
                sample[-1].add_evidence(assign)

            elif node_name == "import":
                file_name = main_node.get_attributes().get_named_item("href").get_node_value()
                points = XMLDialogueReader.extract_dialogue(root_path + "/" + file_name)
                sample.append(points)

        return sample