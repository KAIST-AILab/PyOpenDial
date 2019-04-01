from copy import copy

import pytest

from bn.b_network import BNetwork
from bn.distribs.distribution_builder import CategoricalTableBuilder as CategoricalTableBuilder
from bn.nodes.action_node import ActionNode
from bn.nodes.chance_node import ChanceNode
from bn.values.value_factory import ValueFactory
from datastructs.assignment import Assignment
from test.common.network_examples import NetworkExamples


class TestBNetworkStructure:
    def test_build_basic_network(self):
        bn = NetworkExamples.construct_basic_network()

        assert len(bn.get_nodes()) == 8

        assert len(bn.get_node("Burglary").get_output_nodes()) == 3
        assert len(bn.get_node("Alarm").get_output_nodes()) == 2
        assert len(bn.get_node("Alarm").get_input_nodes()) == 2
        assert len(bn.get_node("Util1").get_input_nodes()) == 2

        assert len(bn.get_node("Burglary").get_values()) == 2
        assert len(bn.get_node("Alarm").get_values()) == 2
        assert len(bn.get_node("MaryCalls").get_values()) == 2

        assert ValueFactory.create(True) in bn.get_node("Burglary").get_values()

        assert bn.get_chance_node("Burglary").get_prob(ValueFactory.create(True)) == pytest.approx(0.001, abs=0.0001)
        assert bn.get_chance_node("Alarm").get_prob(Assignment(["Burglary", "Earthquake"]), ValueFactory.create(True)) == pytest.approx(0.95, abs=0.0001)
        assert bn.get_chance_node("JohnCalls").get_prob(Assignment("Alarm"), ValueFactory.create(True)) == pytest.approx(0.9, abs=0.0001)

        assert len(bn.get_action_node("Action").get_values()) == 3
        assert bn.get_utility_node("Util2").get_utility(Assignment(Assignment("Burglary"), "Action", ValueFactory.create("DoNothing"))) == pytest.approx(-10, abs=0.0001)

    def test_copy(self):
        bn = NetworkExamples.construct_basic_network()

        bn2 = copy(bn)
        b = bn.get_chance_node("Burglary")

        builder = CategoricalTableBuilder("Burglary")
        builder.add_row(ValueFactory.create(True), 0.2)
        builder.add_row(ValueFactory.create(False), 0.8)
        b.set_distrib(builder.build())

        value = bn.get_utility_node("Util1")
        value.add_utility(Assignment(Assignment("Burglary", True), "Action", ValueFactory.create("DoNothing")), -20.0)

        assert len(bn.get_node("Burglary").get_output_nodes()) == 3
        assert len(bn2.get_node("Burglary").get_output_nodes()) == 3

        assert len(bn.get_node("Alarm").get_output_nodes()) == 2
        assert len(bn2.get_node("Alarm").get_output_nodes()) == 2
        assert len(bn.get_node("Alarm").get_input_nodes()) == 2
        assert len(bn2.get_node("Alarm").get_input_nodes()) == 2

        assert len(bn.get_node("Util1").get_input_nodes()) == 2
        assert len(bn2.get_node("Util1").get_input_nodes()) == 2

        assert len(bn2.get_node("Burglary").get_values()) == 2
        assert len(bn2.get_node("Alarm").get_values()) == 2
        assert len(bn2.get_node("MaryCalls").get_values()) == 2

        assert ValueFactory.create(True) in bn2.get_node("Burglary").get_values()

        assert bn2.get_chance_node("Burglary").get_prob(ValueFactory.create(True)) == pytest.approx(0.001, abs=0.0001)
        assert bn2.get_chance_node("Alarm").get_prob(Assignment(["Burglary", "Earthquake"]), ValueFactory.create(True)) == pytest.approx(0.95, abs=0.0001)
        assert bn2.get_chance_node("JohnCalls").get_prob(Assignment("Alarm"), ValueFactory.create(True)) == pytest.approx(0.9, abs=0.0001)

        assert len(bn2.get_action_node("Action").get_values()) == 3
        assert bn2.get_utility_node("Util2").get_utility(Assignment(Assignment("Burglary"), "Action", ValueFactory.create("DoNothing"))) == pytest.approx(-10, abs=0.0001)

    def test_structure(self):
        bn = NetworkExamples.construct_basic_network()

        assert len(bn.get_node("Burglary").get_descendant_ids()) == 5
        assert "Alarm" in bn.get_node("Burglary").get_descendant_ids()
        assert "MaryCalls" in bn.get_node("Burglary").get_descendant_ids()

        assert len(bn.get_node("MaryCalls").get_ancestor_ids()) == 3
        assert "Alarm" in bn.get_node("MaryCalls").get_ancestor_ids()
        assert "Earthquake" in bn.get_node("MaryCalls").get_ancestor_ids()
        assert len(bn.get_node("MaryCalls").get_descendant_ids()) == 0

        assert len(bn.get_node("Util1").get_ancestor_ids()) == 2
        assert "Action" in bn.get_node("Util1").get_ancestor_ids()
        assert len(bn.get_node("Util1").get_descendant_ids()) == 0

        assert len(bn.get_node("Action").get_ancestor_ids()) == 0
        assert "Util1" in bn.get_node("Action").get_descendant_ids()
        assert len(bn.get_node("Action").get_descendant_ids()) == 2

    def test_removal(self):
        bn = NetworkExamples.construct_basic_network()

        bn.remove_node("Earthquake")
        assert len(bn.get_chance_node("Alarm").get_input_nodes()) == 1
        bn.remove_node("Alarm")
        assert len(bn.get_chance_node("MaryCalls").get_input_nodes()) == 0
        assert len(bn.get_chance_node("Burglary").get_output_nodes()) == 2
        assert len(bn.get_nodes()) == 6

        bn = NetworkExamples.construct_basic_network()

        e = bn.get_chance_node("Alarm")
        e.remove_input_node("Earthquake")
        assert len(bn.get_chance_node("Alarm").get_input_nodes()) == 1
        assert len(bn.get_chance_node("Earthquake").get_output_nodes()) == 0

    def test_id_chance(self):
        bn = NetworkExamples.construct_basic_network()

        node = bn.get_node("Alarm")
        node.set_id("Alarm2")
        assert bn.has_node("Alarm2")
        assert not bn.has_node("Alarm")
        assert bn.has_chance_node("Alarm2")
        assert not bn.has_chance_node("Alarm")
        assert "Alarm2" in bn.get_node("Burglary").get_output_node_ids()
        assert "Alarm" not in bn.get_node("Burglary").get_output_node_ids()
        assert "Alarm2" in bn.get_node("MaryCalls").get_input_node_ids()
        assert "Alarm" not in bn.get_node("MaryCalls").get_input_node_ids()

    def test_copy_id_change(self):
        bn = NetworkExamples.construct_basic_network()

        bn2 = copy(bn)
        node = bn.get_node("Earthquake")
        node.set_id("Earthquake2")
        assert "Earthquake2" not in bn2.get_node("Alarm").get_input_node_ids()
        assert "Earthquake2" not in bn2.get_node("Alarm").get_input_node_ids()

        node2 = bn.get_node("Alarm")
        node2.set_id("Alarm2")
        assert "Alarm2" not in bn2.get_node("MaryCalls").get_input_node_ids()
        assert "Alarm2" not in bn2.get_node("Burglary").get_output_node_ids()
        assert "Alarm" in bn2.get_node("Burglary").get_output_node_ids()
        assert "Alarm" in bn2.get_node("MaryCalls").get_input_node_ids()

    def test_table_expansion(self):
        bn = NetworkExamples.construct_basic_network()

        builder = CategoricalTableBuilder("HouseSize")
        builder.add_row(ValueFactory.create("Small"), 0.7)
        builder.add_row(ValueFactory.create("Big"), 0.2)
        builder.add_row(ValueFactory.create("None"), 0.1)

        node = ChanceNode("HouseSize", builder.build())
        bn.add_node(node)
        bn.get_node("Burglary").add_input_node(node)
        assert bn.get_chance_node("Burglary").get_prob(Assignment(["HouseSize", "Small"]), ValueFactory.create(True)) == pytest.approx(0.001, abs=0.0001)
        assert bn.get_chance_node("Burglary").get_prob(Assignment(["HouseSize", "Big"]), ValueFactory.create(True)) == pytest.approx(0.001, abs=0.0001)
        bn.get_node("Alarm").add_input_node(node)
        assert bn.get_chance_node("Alarm").get_prob(Assignment(["Burglary", "Earthquake"]), ValueFactory.create(True)) == pytest.approx(0.95, abs=0.0001)
        assert bn.get_chance_node("Alarm").get_prob(Assignment(Assignment(["Burglary", "Earthquake"]), "HouseSize", ValueFactory.create("None")), ValueFactory.create(True)) == pytest.approx(0.95, abs=0.0001)

    def test_default_value(self):
        builder = CategoricalTableBuilder("Burglary")
        builder.add_row(ValueFactory.create(False), 0.8)
        assert builder.build().get_prob(ValueFactory.none()) == pytest.approx(0.199, abs=0.01)
        builder.remove_row(ValueFactory.create(False))
        assert builder.build().get_prob(ValueFactory.none()) == pytest.approx(0.999, abs=0.01)
        # assert node.hasProb(Assignment(), ValueFactory.none())
        builder = CategoricalTableBuilder("Burglary")
        builder.add_row(ValueFactory.create(False), 0.999)
        assert builder.build().get_prob(ValueFactory.none()) == pytest.approx(0.0, abs=0.01)
        # assert not node.hasProb(Assignment(), ValueFactory.none())

    def test_sorted_nodes(self):
        bn = NetworkExamples.construct_basic_network()
        assert "Action" == bn.get_sorted_nodes()[7].get_id()
        assert "Burglary" == bn.get_sorted_nodes()[6].get_id()
        assert "Earthquake" == bn.get_sorted_nodes()[5].get_id()
        assert "Alarm" == bn.get_sorted_nodes()[4].get_id()
        assert "Util1" == bn.get_sorted_nodes()[3].get_id()
        assert "Util2" == bn.get_sorted_nodes()[2].get_id()
        assert "JohnCalls" == bn.get_sorted_nodes()[1].get_id()
        assert "MaryCalls" == bn.get_sorted_nodes()[0].get_id()

        d1 = ActionNode("a_m'")
        d2 = ActionNode("a_m.obj'")
        d3 = ActionNode("a_m.place'")
        bn2 = BNetwork()
        bn2.add_node(d1)
        bn2.add_node(d2)
        bn2.add_node(d3)
        assert "a_m'" == bn2.get_sorted_nodes()[2].get_id()
        assert "a_m.obj'" == bn2.get_sorted_nodes()[1].get_id()
        assert "a_m.place'" == bn2.get_sorted_nodes()[0].get_id()

    def test_cliques(self):
        bn = NetworkExamples.construct_basic_network()
        assert len(bn.get_cliques()) == 1
        assert len(bn.get_cliques()[0]) == 8

        bn.get_node("JohnCalls").remove_input_node("Alarm")
        cliques = bn.get_cliques()
        assert len(cliques) == 2
        # 원래 테스트 케이스와 셋에 대한 정렬 순서가 달라 테스트케이스 그대로 활용 불가능.
        # assert len(bn.get_cliques()[1]) == 7
        # assert len(bn.get_cliques()[0]) == 1
        result = dict()
        for clique in cliques:
            if len(clique) not in result:
                result[len(clique)] = 0

            result[len(clique)] += 1

        assert result[7] == 1
        del result[7]
        assert result[1] == 1
        del result[1]

        assert len(result) == 0

        bn.get_node("Alarm").remove_input_node("Burglary")
        bn.get_node("Alarm").remove_input_node("Earthquake")
        cliques = bn.get_cliques()
        assert len(cliques) == 4
        # 원래 테스트 케이스와 셋에 대한 정렬 순서가 달라 테스트케이스 그대로 활용 불가능.
        # assert len(bn.get_cliques()[3]) == 2
        # assert len(bn.get_cliques()[2]) == 4
        # assert len(bn.get_cliques()[1]) == 1
        # assert len(bn.get_cliques()[0]) == 1

        result = dict()
        for clique in cliques:
            if len(clique) not in result:
                result[len(clique)] = 0

            result[len(clique)] += 1

        assert result[1] == 2
        del result[1]
        assert result[2] == 1
        del result[2]
        assert result[4] == 1
        del result[4]

        assert len(result) == 0
