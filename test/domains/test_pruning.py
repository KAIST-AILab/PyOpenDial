from copy import copy

from sortedcontainers import SortedSet

from bn.distribs.distribution_builder import CategoricalTableBuilder
from bn.values.value_factory import ValueFactory
from dialogue_system import DialogueSystem
from readers.xml_domain_reader import XMLDomainReader
from test.common.inference_checks import InferenceChecks


class TestPruning:
    domainFile = "test/data/domain1.xml"

    domain = XMLDomainReader.extract_domain(domainFile)
    inference = InferenceChecks()
    InferenceChecks.exact_threshold = 0.1
    InferenceChecks.sampling_threshold = 0.1
    system = DialogueSystem(domain)
    system.get_settings().show_gui = False

    system.start_system()

    def test_pruning0(self):
        assert len(TestPruning.system.get_state().get_node_ids()) == 15
        assert len(TestPruning.system.get_state().get_evidence().get_variables()) == 0

    def test_pruning1(self):
        TestPruning.inference.check_prob(TestPruning.system.get_state(), "a_u", "Greeting", 0.8)
        TestPruning.inference.check_prob(TestPruning.system.get_state(), "a_u", "None", 0.2)

    def test_pruning2(self):
        TestPruning.inference.check_prob(TestPruning.system.get_state(), "i_u", "Inform", 0.7 * 0.8)
        TestPruning.inference.check_prob(TestPruning.system.get_state(), "i_u", "None", 1 - 0.7 * 0.8)

    def test_pruning3(self):
        TestPruning.inference.check_prob(TestPruning.system.get_state(), "direction", "straight", 0.79)
        TestPruning.inference.check_prob(TestPruning.system.get_state(), "direction", "left", 0.20)
        TestPruning.inference.check_prob(TestPruning.system.get_state(), "direction", "right", 0.01)

    def test_pruning4(self):
        TestPruning.inference.check_prob(TestPruning.system.get_state(), "o", "and we have var1=value2", 0.3)
        TestPruning.inference.check_prob(TestPruning.system.get_state(), "o", "and we have localvar=value1", 0.2)
        TestPruning.inference.check_prob(TestPruning.system.get_state(), "o", "and we have localvar=value3", 0.31)

    def test_pruning5(self):
        TestPruning.inference.check_prob(TestPruning.system.get_state(), "o2", "here is value1", 0.35)
        TestPruning.inference.check_prob(TestPruning.system.get_state(), "o2", "and value2 is over there", 0.07)
        TestPruning.inference.check_prob(TestPruning.system.get_state(), "o2", "value3, finally", 0.28)

    def test_pruning6(self):
        initial_state = copy(TestPruning.system.get_state())

        builder = CategoricalTableBuilder("var1")
        builder.add_row("value2", 0.9)
        TestPruning.system.get_state().add_to_state(builder.build())

        TestPruning.inference.check_prob(TestPruning.system.get_state(), "o", "and we have var1=value2", 0.3)
        TestPruning.inference.check_prob(TestPruning.system.get_state(), "o", "and we have localvar=value1", 0.2)
        TestPruning.inference.check_prob(TestPruning.system.get_state(), "o", "and we have localvar=value3", 0.31)

        TestPruning.system.get_state().reset(initial_state)

    def test_pruning7(self):
        TestPruning.inference.check_prob(TestPruning.system.get_state(), "a_u2", "[Greet, HowAreYou]", 0.7)
        TestPruning.inference.check_prob(TestPruning.system.get_state(), "a_u2", "none", 0.1)
        TestPruning.inference.check_prob(TestPruning.system.get_state(), "a_u2", "[HowAreYou]", 0.2)

    def test_pruning8(self):
        initial_state = copy(TestPruning.system.get_state())

        created_nodes = SortedSet()
        for node_id in TestPruning.system.get_state().get_node_ids():
            if node_id.find("a_u3^") != -1:
                created_nodes.add(node_id)

        assert len(created_nodes) == 2

        values = TestPruning.system.get_state().get_node(created_nodes[0] + "").get_values()
        if ValueFactory.create("Greet") in values:
            greet_node = created_nodes[0]  # created_nodes.first()
            howareyou_node = created_nodes[-1]  # created_nodes.last()
        else:
            greet_node = created_nodes[-1]  # created_nodes.last()
            howareyou_node = created_nodes[0]  # created_nodes.first()

        TestPruning.inference.check_prob(TestPruning.system.get_state(), "a_u3", "[" + howareyou_node + "," + greet_node + "]", 0.7)
        TestPruning.inference.check_prob(TestPruning.system.get_state(), "a_u3", "none", 0.1)
        TestPruning.inference.check_prob(TestPruning.system.get_state(), "a_u3", "[" + howareyou_node + "]", 0.2)
        TestPruning.inference.check_prob(TestPruning.system.get_state(), greet_node + "", "Greet", 0.7)
        TestPruning.inference.check_prob(TestPruning.system.get_state(), howareyou_node + "", "HowAreYou", 0.9)

        TestPruning.system.get_state().reset(initial_state)
