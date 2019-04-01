from bn.distribs.distribution_builder import CategoricalTableBuilder
from dialogue_system import DialogueSystem
from readers.xml_domain_reader import XMLDomainReader
from test.common.inference_checks import InferenceChecks


class TestPlanning:
    domain_file = "test/data/domain3.xml"
    domain_file2 = "test/data/basicplanning.xml"
    domain_file3 = "test/data/planning2.xml"
    settings_file = "test/data/settings_test2.xml"

    inference = InferenceChecks()
    domain = XMLDomainReader.extract_domain(domain_file)
    domain2 = XMLDomainReader.extract_domain(domain_file2)
    domain3 = XMLDomainReader.extract_domain(domain_file3)

    def test_planning(self):
        system = DialogueSystem(TestPlanning.domain)

        system.get_settings().show_gui = False

        system.start_system()
        assert len(system.get_state().get_nodes()) == 3
        assert len(system.get_state().get_chance_nodes()) == 3
        assert len(system.get_state().get_evidence().get_variables()) == 0
        TestPlanning.inference.check_prob(system.get_state(), "a_m3", "Do", 1.0)
        TestPlanning.inference.check_prob(system.get_state(), "obj(a_m3)", "A", 1.0)

    def test_planning2(self):
        system = DialogueSystem(TestPlanning.domain2)

        system.get_settings().show_gui = False

        system.start_system()
        assert len(system.get_state().get_node_ids()) == 2
        assert not system.get_state().has_chance_node("a_m")

    def test_planning3(self):
        system = DialogueSystem(TestPlanning.domain2)

        system.get_settings().show_gui = False

        system.get_settings().horizon = 2
        system.start_system()
        TestPlanning.inference.check_prob(system.get_state(), "a_m", "AskRepeat", 1.0)

    def test_planning4(self):
        system = DialogueSystem(TestPlanning.domain3)

        system.get_settings().show_gui = False

        system.get_settings().horizon = 3
        system.start_system()

        t1 = CategoricalTableBuilder("a_u")
        t1.add_row("Ask(Coffee)", 0.95)
        t1.add_row("Ask(Tea)", 0.02)
        system.add_content(t1.build())
        TestPlanning.inference.check_prob(system.get_state(), "a_m", "Do(Coffee)", 1.0)

    def test_planning5(self):
        system = DialogueSystem(TestPlanning.domain3)

        system.get_settings().show_gui = False

        system.get_settings().horizon = 3
        system.start_system()

        t1 = CategoricalTableBuilder("a_u")
        t1.add_row("Ask(Coffee)", 0.3)
        t1.add_row("Ask(Tea)", 0.3)
        system.add_content(t1.build())

        TestPlanning.inference.check_prob(system.get_state(), "a_m", "AskRepeat", 1.0)
