import pytest

from bn.values.value_factory import ValueFactory
from dialogue_system import DialogueSystem
from domains.rules.effects.basic_effect import BasicEffect
from domains.rules.effects.effect import Effect
from modules.forward_planner import ForwardPlanner
from modules.state_pruner import StatePruner
from readers.xml_domain_reader import XMLDomainReader
from test.common.inference_checks import InferenceChecks


class TestRule3:
    domain_file = "test/data/rulepriorities.xml"
    test1_domain_file = "test/data/domain5.xml"
    test2_domain_file = "test/data/domainthesis.xml"
    predict_domain_file = "test/data/prediction.xml"
    incondition_file = "test/data/incondition.xml"

    def test_priority(self):
        system = DialogueSystem(XMLDomainReader.extract_domain(TestRule3.domain_file))
        system.get_settings().show_gui = False
        system.start_system()
        assert system.get_content("a_u").get_prob("Opening") == pytest.approx(0.8, abs=0.01)
        assert system.get_content("a_u").get_prob("Nothing") == pytest.approx(0.1, abs=0.01)
        assert system.get_content("a_u").get_prob("start") == pytest.approx(0.0, abs=0.01)
        assert not system.get_content("a_u").to_discrete().has_prob(ValueFactory.create("start"))

    def test_1(self):
        domain = XMLDomainReader.extract_domain(TestRule3.test1_domain_file)
        inference = InferenceChecks()
        inference.exact_threshold = 0.06
        system = DialogueSystem(domain)
        system.get_settings().show_gui = False
        system.detach_module(ForwardPlanner)
        StatePruner.enable_reduction = False
        system.start_system()

        inference.check_prob(system.get_state(), "found", "A", 0.7)

        inference.check_prob(system.get_state(), "found2", "D", 0.3)
        inference.check_prob(system.get_state(), "found2", "C", 0.5)

        StatePruner.enable_reduction = True

    def test_2(self):
        inference = InferenceChecks()

        domain = XMLDomainReader.extract_domain(TestRule3.test2_domain_file)
        system = DialogueSystem(domain)
        system.get_settings().show_gui = False

        system.detach_module(ForwardPlanner)
        StatePruner.enable_reduction = False
        system.start_system()

        inference.check_prob(system.get_state(), "graspable(obj1)", "True", 0.81)

        inference.check_prob(system.get_state(), "graspable(obj2)", "True", 0.16)
        inference.check_util(system.get_state(), "a_m'", "grasp(obj1)", 0.592)

        StatePruner.enable_reduction = True

    def test_outputs(self):
        effects = []

        assert Effect(effects) == Effect.parse_effect("Void")
        effects.append(BasicEffect("v1", "val1"))
        assert Effect(effects) == Effect.parse_effect("v1:=val1")

        effects.append(BasicEffect("v2", ValueFactory.create("val2"), 1, False, False))
        assert Effect(effects) == Effect.parse_effect("v1:=val1 ^ v2+=val2")

        effects.append(BasicEffect("v2", ValueFactory.create("val3"), 1, True, True))
        assert Effect(effects) == Effect.parse_effect("v1:=val1 ^ v2+=val2 ^ v2!=val3")

    def test_incondition(self):
        domain = XMLDomainReader.extract_domain(TestRule3.incondition_file)

        system = DialogueSystem(domain)
        system.get_settings().show_gui = False

        system.start_system()

        assert system.get_content("out").get_prob("val1 is in [val1, val2]") + system.get_content("out").get_prob("val1 is in [val2, val1]") == pytest.approx(0.56, abs=0.01)
        assert system.get_content("out2").get_prob("this is a string is matched") == pytest.approx(0.5, abs=0.01)
