import pytest

from bn.distribs.distribution_builder import CategoricalTableBuilder
from dialogue_system import DialogueSystem
from domains.rules.distribs.anchored_rule import AnchoredRule
from readers.xml_domain_reader import XMLDomainReader


class TestRuleAndParams:
    domain_file = "test/data/rulesandparams.xml"

    def test_rule_and_params(self):
        domain = XMLDomainReader.extract_domain(TestRuleAndParams.domain_file)
        system = DialogueSystem(domain)

        system.get_settings().show_gui = False

        system.start_system()
        assert system.get_content("theta_moves").to_continuous().get_function().get_mean()[0] == pytest.approx(0.2, abs=0.02)
        assert system.get_content("a_u^p").get_prob("I want left") == pytest.approx(0.12, abs=0.03)
        assert len(system.get_state().get_chance_node("theta_moves").get_output_node_ids()) == 1
        assert system.get_state().has_chance_node("movements")
        assert isinstance(system.get_state().get_chance_node("movements").get_distrib(), AnchoredRule)

        t = CategoricalTableBuilder("a_u")
        t.add_row("I want left", 0.8)
        t.add_row("I want forward", 0.1)
        system.add_content(t.build())

        assert len(system.get_state().get_chance_node("theta_moves").get_output_node_ids()) == 0
        assert not system.get_state().has_chance_node("movements")
        assert system.get_content("theta_moves").to_continuous().get_function().get_mean()[0] == pytest.approx(2.0 / 6.0, abs=0.07)

        system.add_content("a_m", "turning left")

        assert system.get_content("a_u^p").get_prob("I want left") == pytest.approx(0.23, abs=0.04)
        assert len(system.get_state().get_chance_node("theta_moves").get_output_node_ids()) == 1
