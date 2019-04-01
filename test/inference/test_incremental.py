from time import sleep

import pytest

from bn.distribs.distribution_builder import CategoricalTableBuilder
from bn.distribs.single_value_distribution import SingleValueDistribution
from bn.values.value_factory import ValueFactory
from dialogue_system import DialogueSystem
from readers.xml_domain_reader import XMLDomainReader


class TestIncremental:
    domain_file = "test/data/incremental-domain.xml"

    def test1(self):
        domain = XMLDomainReader.extract_domain(TestIncremental.domain_file)
        system = DialogueSystem(domain)

        # NEED GUI & Recording
        system.get_settings().show_gui = False
        # system.get_settings().recording = Settings.Recording.ALL

        system.start_system()
        system.add_content(system.get_settings().user_speech, "busy")
        system.add_incremental_content(SingleValueDistribution("u_u", "go"), False)

        sleep(0.1)

        assert ValueFactory.create("go") in system.get_content("u_u").get_values()

        t = CategoricalTableBuilder("u_u")
        t.add_row("forward", 0.7)
        t.add_row("backward", 0.2)

        system.add_incremental_content(t.build(), True)

        sleep(0.1)

        assert ValueFactory.create("go forward") in system.get_content("u_u").get_values()
        assert system.get_content("u_u").get_prob("go backward") == pytest.approx(0.2, abs=0.001)
        assert system.get_state().has_chance_node("nlu")

        system.add_content(system.get_settings().user_speech, "None")
        assert len(system.get_state().get_chance_nodes()) == 7
        system.add_incremental_content(SingleValueDistribution("u_u", "please"), True)
        assert system.get_state().get_evidence().contains_pair("=_a_u", ValueFactory.create(True))

        assert system.get_content("u_u").get_prob("go please") == pytest.approx(0.1, abs=0.001)
        assert system.get_state().has_chance_node("nlu")

        system.get_state().set_as_committed("u_u")
        assert not system.get_state().has_chance_node("nlu")

        t2 = CategoricalTableBuilder("u_u")
        t2.add_row("I said go backward", 0.3)

        system.add_incremental_content(t2.build(), True)
        assert system.get_content("a_u").get_prob("Request(Backward)") == pytest.approx(0.82, abs=0.05)
        assert ValueFactory.create("I said go backward") in system.get_content("u_u").get_values()
        assert system.get_state().has_chance_node("nlu")

        system.get_state().set_as_committed("u_u")
        assert not system.get_state().has_chance_node("nlu")
        system.add_incremental_content(SingleValueDistribution("u_u", "yes that is right"), False)
        assert ValueFactory.create("yes that is right") in system.get_content("u_u").get_values()
