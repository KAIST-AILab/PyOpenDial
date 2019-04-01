from bn.distribs.distribution_builder import CategoricalTableBuilder as CategoricalTableBuilder
from dialogue_system import DialogueSystem
from modules.forward_planner import ForwardPlanner
from modules.state_pruner import StatePruner
from readers.xml_domain_reader import XMLDomainReader
from test.common.inference_checks import InferenceChecks


class TestRule1:
    domain = XMLDomainReader.extract_domain('test/data/domain1.xml')
    inference = InferenceChecks()

    def test_1(self):
        system = DialogueSystem(TestRule1.domain)
        system.detach_module(ForwardPlanner)
        StatePruner.enable_reduction = False
        system.get_settings().show_gui = False
        system.start_system()

        TestRule1.inference.check_prob(system.get_state(), "a_u", "Greeting", 0.8)
        TestRule1.inference.check_prob(system.get_state(), "a_u", "None", 0.2)

        StatePruner.enable_reduction = True

    def test_2(self):
        system = DialogueSystem(TestRule1.domain)
        system.detach_module(ForwardPlanner)
        StatePruner.enable_reduction = False
        system.get_settings().show_gui = False
        system.start_system()

        TestRule1.inference.check_prob(system.get_state(), "i_u", "Inform", 0.7 * 0.8)
        TestRule1.inference.check_prob(system.get_state(), "i_u", "None", 1. - 0.7 * 0.8)

        StatePruner.enable_reduction = True

    def test_3(self):
        InferenceChecks.exact_threshold = 0.06

        system = DialogueSystem(TestRule1.domain)
        system.detach_module(ForwardPlanner)
        StatePruner.enable_reduction = False
        system.get_settings().show_gui = False
        system.start_system()

        TestRule1.inference.check_prob(system.get_state(), "direction", "straight", 0.79)
        TestRule1.inference.check_prob(system.get_state(), "direction", "left", 0.20)
        TestRule1.inference.check_prob(system.get_state(), "direction", "right", 0.01)

        StatePruner.enable_reduction = True

    def test_4(self):
        system = DialogueSystem(TestRule1.domain)
        system.detach_module(ForwardPlanner)
        StatePruner.enable_reduction = False
        system.get_settings().show_gui = False
        system.start_system()

        TestRule1.inference.check_prob(system.get_state(), "o", "and we have var1=value2", 0.3)
        TestRule1.inference.check_prob(system.get_state(), "o", "and we have localvar=value1", 0.2)
        TestRule1.inference.check_prob(system.get_state(), "o", "and we have localvar=value3", 0.28)

        StatePruner.enable_reduction = True

    def test_5(self):
        system = DialogueSystem(TestRule1.domain)
        system.detach_module(ForwardPlanner)
        StatePruner.enable_reduction = False
        system.get_settings().show_gui = False
        system.start_system()

        TestRule1.inference.check_prob(system.get_state(), "o2", "here is value1", 0.35)
        TestRule1.inference.check_prob(system.get_state(), "o2", "and value2 is over there", 0.07)
        TestRule1.inference.check_prob(system.get_state(), "o2", "value3, finally", 0.28)

        StatePruner.enable_reduction = True

    def test_6(self):
        InferenceChecks.exact_threshold = 0.06

        system = DialogueSystem(TestRule1.domain)
        system.detach_module(ForwardPlanner)
        system.get_settings().show_gui = False
        system.start_system()

        builder = CategoricalTableBuilder("var1")
        builder.add_row("value2", 0.9)
        system.add_content(builder.build())

        TestRule1.inference.check_prob(system.get_state(), "o", "and we have var1=value2", 0.9)
        TestRule1.inference.check_prob(system.get_state(), "o", "and we have localvar=value1", 0.05)
        TestRule1.inference.check_prob(system.get_state(), "o", "and we have localvar=value3", 0.04)

        StatePruner.enable_reduction = True

    def test_7(self):
        system = DialogueSystem(TestRule1.domain)
        system.detach_module(ForwardPlanner)
        StatePruner.enable_reduction = False
        system.get_settings().show_gui = False
        system.start_system()

        TestRule1.inference.check_prob(system.get_state(), "a_u2", "[Greet, HowAreYou]", 0.7)
        TestRule1.inference.check_prob(system.get_state(), "a_u2", "[]", 0.1)
        TestRule1.inference.check_prob(system.get_state(), "a_u2", "[HowAreYou]", 0.2)

        StatePruner.enable_reduction = True
