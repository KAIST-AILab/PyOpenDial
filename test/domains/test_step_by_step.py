import pytest

from bn.distribs.distribution_builder import CategoricalTableBuilder
from datastructs.assignment import Assignment
from dialogue_system import DialogueSystem
from modules.forward_planner import ForwardPlanner
from readers.xml_domain_reader import XMLDomainReader


class TestStepByStep:
    def test_domain(self):
        system = DialogueSystem(XMLDomainReader.extract_domain("test/data/example-step-by-step_params.xml"))
        system.detach_module(ForwardPlanner)
        system.get_settings().show_gui = False
        system.start_system()

        o1 = CategoricalTableBuilder("u_u")
        o1.add_row("move a little bit left", 0.4)
        o1.add_row("please move a little right", 0.3)
        system.add_content(o1.build())
        assert system.get_state().query_util(["a_m'"]).get_util(Assignment("a_m'", "AskRepeat")) == pytest.approx(0.0, abs=0.3)
        assert system.get_state().query_util(["a_m'"]).get_util(Assignment("a_m'", "Move(Left)")) == pytest.approx(-0.1, abs=0.15)

        system.get_state().remove_nodes(system.get_state().get_action_node_ids())
        system.get_state().remove_nodes(system.get_state().get_utility_node_ids())

        o1 = CategoricalTableBuilder("u_u")
        o1.add_row("move a little bit left", 0.5)
        o1.add_row("please move a little left", 0.2)
        system.add_content(o1.build())

        assert system.get_state().query_util(["a_m'"]).get_util(Assignment("a_m'", "AskRepeat")) == pytest.approx(0.0, abs=0.3)
        assert system.get_state().query_util(["a_m'"]).get_util(Assignment("a_m'", "Move(Left)")) == pytest.approx(0.2, abs=0.15)

        system.get_state().remove_nodes(system.get_state().get_action_node_ids())
        system.get_state().remove_nodes(system.get_state().get_utility_node_ids())

        o1 = CategoricalTableBuilder("u_u")
        o1.add_row("now move right please", 0.8)
        system.add_content(o1.build())

        assert system.get_state().query_util(["a_m'"]).get_util(Assignment("a_m'", "AskRepeat")) == pytest.approx(0.0, abs=0.3)
        assert system.get_state().query_util(["a_m'"]).get_util(Assignment("a_m'", "Move(Right)")) == pytest.approx(0.3, abs=0.15)

        system.get_state().remove_nodes(system.get_state().get_action_node_ids())
        system.get_state().remove_nodes(system.get_state().get_utility_node_ids())

        o1 = CategoricalTableBuilder("u_u")
        o1.add_row("move left", 0.7)
        system.add_content(o1.build())

        assert system.get_state().query_util(["a_m'"]).get_util(Assignment("a_m'", "AskRepeat")) == pytest.approx(0.0, abs=0.3)
        assert system.get_state().query_util(["a_m'"]).get_util(Assignment("a_m'", "Move(Left)")) == pytest.approx(0.2, abs=0.15)

        system.get_state().remove_nodes(system.get_state().get_action_node_ids())
        system.get_state().remove_nodes(system.get_state().get_utility_node_ids())

        o1 = CategoricalTableBuilder("u_u")
        o1.add_row("turn left", 0.32)
        o1.add_row("move left again", 0.3)
        system.add_content(o1.build())

        assert system.get_state().query_util(["a_m'"]).get_util(Assignment("a_m'", "AskRepeat")) == pytest.approx(0.0, abs=0.3)
        assert system.get_state().query_util(["a_m'"]).get_util(Assignment("a_m'", "Move(Left)")) == pytest.approx(0.1, abs=0.15)

        system.get_state().remove_nodes(system.get_state().get_action_node_ids())
        system.get_state().remove_nodes(system.get_state().get_utility_node_ids())

        o1 = CategoricalTableBuilder("u_u")
        o1.add_row("and do that again", 0.0)
        system.add_content(o1.build())

        system.get_state().remove_nodes(system.get_state().get_action_node_ids())
        system.get_state().remove_nodes(system.get_state().get_utility_node_ids())

        o1 = CategoricalTableBuilder("u_u")
        o1.add_row("turn left", 1.0)
        system.add_content(o1.build())

        assert system.get_state().query_util(["a_m'"]).get_util(Assignment("a_m'", "AskRepeat")) == pytest.approx(0.0, abs=0.3)
        assert system.get_state().query_util(["a_m'"]).get_util(Assignment("a_m'", "Move(Left)")) == pytest.approx(0.5, abs=0.15)

        system.get_state().remove_nodes(system.get_state().get_action_node_ids())
        system.get_state().remove_nodes(system.get_state().get_utility_node_ids())

        o1 = CategoricalTableBuilder("u_u")
        o1.add_row("turn right", 0.4)
        system.add_content(o1.build())

        assert system.get_state().query_util(["a_m'"]).get_util(Assignment("a_m'", "AskRepeat")) == pytest.approx(0.0, abs=0.3)
        assert system.get_state().query_util(["a_m'"]).get_util(Assignment("a_m'", "Move(Right)")) == pytest.approx(-0.1, abs=0.15)

        system.get_state().remove_nodes(system.get_state().get_action_node_ids())
        system.get_state().remove_nodes(system.get_state().get_utility_node_ids())

        o1 = CategoricalTableBuilder("u_u")
        o1.add_row("please turn right", 0.8)
        system.add_content(o1.build())

        assert system.get_state().query_util(["a_m'"]).get_util(Assignment("a_m'", "AskRepeat")) == pytest.approx(0.0, abs=0.3)
        assert system.get_state().query_util(["a_m'"]).get_util(Assignment("a_m'", "Move(Right)")) == pytest.approx(0.3, abs=0.15)

        system.get_state().remove_nodes(system.get_state().get_action_node_ids())
        system.get_state().remove_nodes(system.get_state().get_utility_node_ids())

        o1 = CategoricalTableBuilder("u_u")
        o1.add_row("and turn a bit left", 0.3)
        o1.add_row("move a bit left", 0.3)
        system.add_content(o1.build())

        assert system.get_state().query_util(["a_m'"]).get_util(Assignment("a_m'", "AskRepeat")) == pytest.approx(0.0, abs=0.3)
        assert system.get_state().query_util(["a_m'"]).get_util(Assignment("a_m'", "Move(Left)")) == pytest.approx(0.1, abs=0.15)

        system.get_state().remove_nodes(system.get_state().get_action_node_ids())
        system.get_state().remove_nodes(system.get_state().get_utility_node_ids())
