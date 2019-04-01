import io

import numpy as np

from bn.b_network import BNetwork
from bn.distribs.continuous_distribution import ContinuousDistribution
from bn.distribs.density_functions.dirichlet_density_function import DirichletDensityFunction
from bn.distribs.density_functions.gaussian_density_function import GaussianDensityFunction
from bn.distribs.density_functions.uniform_density_function import UniformDensityFunction
from bn.distribs.distribution_builder import CategoricalTableBuilder as CategoricalTableBuilder
from bn.nodes.chance_node import ChanceNode
from bn.values.value_factory import ValueFactory
from utils.xml_utils import XMLUtils


class XMLStateReader:
    @staticmethod
    def extract_bayesian_network(file, tag):
        """
        Returns the initial state or parameters from the XML document, for the given
        domain (where the variable types are already declared)

        :param file: the file to process
        :param tag: the XML tag to search for
        :return: the specified Bayesian network
        """
        # extract the XML document
        document = XMLUtils.get_xml_document(file)
        main_node = XMLUtils.get_main_node(document)

        if main_node.tag == tag:
            return XMLStateReader.get_bayesian_network(main_node)

        for child_node in main_node:
            if child_node.tag == tag:
                return XMLStateReader.get_bayesian_network(child_node)

        raise ValueError()

    @staticmethod
    def extract_bayesian_network_from_string(full_string):
        """
        Extracts the bayesian network from a XML string.

        :param full_string: the string containing the initial state content
        :return: the corresponding Bayesian network
        """
        # extract the XML document
        document = XMLUtils.get_xml_document(io.StringIO(full_string))
        main_node = XMLUtils.get_main_node(document)

        if main_node.tag == 'state':
            return XMLStateReader.get_bayesian_network(main_node)

        for child_node in main_node:
            if child_node.tag == 'state':
                return XMLStateReader.get_bayesian_network(child_node)

        return BNetwork()

    @staticmethod
    def get_bayesian_network(main_node):
        """
        Returns the initial state or parameters from the XML document, for the given
        domain (where the variable types are already declared)

        :param main_node: the main node for the XML document
        :return: the corresponding dialogue state
        """
        state = BNetwork()

        for child_node in main_node:
            if child_node.tag == 'variable':
                chance_node = XMLStateReader.create_chance_node(child_node)
                state.add_node(chance_node)
            elif child_node.tag != '#text' and child_node.tag != '#comment':
                raise ValueError()

        return state

    @staticmethod
    def create_chance_node(node):
        """
        Creates a new chance node corresponding to the XML specification

        :param node: the XML node
        :return: the resulting chance node encoded
        """
        if len(node.attrib) == 0:
            raise ValueError()

        try:
            label = node.attrib['id'].strip()
        except:
            raise ValueError()

        if len(label) == 0:
            raise ValueError()

        builder = CategoricalTableBuilder(label)
        distrib = None

        for child_node in node:
            if child_node.tag == 'value':
                # first case: the chance node is described as a categorical table.
                # extracting the value
                prob = XMLStateReader._get_probability(child_node)
                value = ValueFactory.create(child_node.text.strip())
                builder.add_row(value, prob)
            elif child_node.tag == 'distrib':
                # second case: the chance node is described by a parametric continuous distribution.
                try:
                    distrib_type = child_node.attrib['type'].lower()

                    if distrib_type == 'gaussian':
                        distrib = ContinuousDistribution(label, XMLStateReader._get_gaussian(child_node))
                    elif distrib_type == 'uniform':
                        distrib = ContinuousDistribution(label, XMLStateReader._get_uniform(child_node))
                    elif distrib_type == 'dirichlet':
                        distrib = ContinuousDistribution(label, XMLStateReader._get_dirichlet(child_node))
                except:
                    raise ValueError()

        if distrib is not None:
            return ChanceNode(label, distrib)

        total_prob = builder.get_total_prob()
        # TODO: check eps
        eps = 1e-8
        if total_prob > 1.0 + eps:
            raise ValueError()

        return ChanceNode(label, builder.build())

    @staticmethod
    def _get_probability(node):
        """
        Returns the probability of the value defined in the XML node (default to 1.0f
        is none is declared)
        :param node: the XML node
        :return: the value probability
        """
        prob = 1.

        try:
            prob = node.attrib['prob']
        except:
            return prob

        return float(prob)

    @staticmethod
    def _get_gaussian(node):
        """
        Extracts the gaussian density function described by the XML specification

        :param node: the XML node
        :return: the corresponding Gaussian PDF properly encoded
        """
        mean = None
        variance = None

        for child_node in node:
            str_value = child_node.text.strip()
            if str_value[:1] == '[':
                value = ValueFactory.create(str_value)
            else:
                value = ValueFactory.create("[%s]" % str_value)

            if child_node.tag == 'mean':
                mean = value.get_array()
            elif child_node.tag == 'variance':
                variance = value.get_array()
            else:
                raise ValueError()

        if mean is None or variance is None:
            raise ValueError()

        return GaussianDensityFunction(mean, variance)

    @staticmethod
    def _get_uniform(node):
        """
        Extracts the uniform density function described by the XML specification

        :param node: the XML node
        :return: the corresponding uniform PDF properly encoded
        """
        min_val = None
        max_val = None

        for child_node in node:
            value = float(child_node.text.strip())
            if child_node.tag == 'min':
                min_val = value
            elif child_node.tag == 'max':
                max_val = value
            else:
                raise ValueError()

        if min_val is None or max_val is None:
            raise ValueError()

        return UniformDensityFunction(min_val, max_val)

    @staticmethod
    def _get_dirichlet(node):
        """
        Extracts the Dirichlet density function described by the XML specification

        :param node: the XML node
        :return: the corresponding Dirichlet PDF properly encoded
        """
        alpha_list = list()

        for child in node:
            if child.tag == 'alpha':
                alpha = float(child.text.strip())
                alpha_list.append(alpha)
            else:
                raise ValueError()

        if len(alpha_list) == 0:
            raise ValueError()

        return DirichletDensityFunction(np.array(alpha_list))
