from copy import copy

import pytest

from bn.distribs.single_value_distribution import SingleValueDistribution
from datastructs.assignment import Assignment
from inference.approximate.sampling_algorithm import SamplingAlgorithm
from inference.exact.naive_inference import NaiveInference
from inference.exact.variable_elimination import VariableElimination
from inference.switching_algorithm import SwitchingAlgorithm
from test.common.network_examples import NetworkExamples


class TestNetworkReduction:
    network = NetworkExamples.construct_basic_network2()
    ve = VariableElimination()

    # change the variable name 'is' -> 'iz'
    iz = SamplingAlgorithm(3000, 500)
    naive = NaiveInference()
    sw = SwitchingAlgorithm()

    def test1(self):
        reduced_net = TestNetworkReduction.ve.reduce(TestNetworkReduction.network, ["Burglary", "Earthquake", "MaryCalls"])
        reduced_net2 = TestNetworkReduction.naive.reduce(TestNetworkReduction.network, ["Burglary", "Earthquake", "MaryCalls"])
        reduced_net3 = TestNetworkReduction.iz.reduce(TestNetworkReduction.network, ["Burglary", "Earthquake", "MaryCalls"])
        reduced_net4 = TestNetworkReduction.sw.reduce(TestNetworkReduction.network, ["Burglary", "Earthquake", "MaryCalls"])

        assert len(reduced_net.get_nodes()) == 3
        assert len(reduced_net2.get_nodes()) == 3
        assert len(reduced_net3.get_nodes()) == 3
        assert len(reduced_net4.get_nodes()) == 3
        assert TestNetworkReduction.ve.query_prob(TestNetworkReduction.network, ["MaryCalls"], Assignment("Burglary")).get_prob(Assignment("MaryCalls")) == pytest.approx(TestNetworkReduction.ve.query_prob(reduced_net, ["MaryCalls"], Assignment("Burglary")).get_prob(Assignment("MaryCalls")),
                                                                                                                                                                      abs=0.0001)
        assert TestNetworkReduction.ve.query_prob(TestNetworkReduction.network, ["MaryCalls"], Assignment("Burglary")).get_prob(Assignment("MaryCalls")) == pytest.approx(TestNetworkReduction.ve.query_prob(reduced_net2, ["MaryCalls"], Assignment("Burglary")).get_prob(Assignment("MaryCalls")),
                                                                                                                                                                      abs=0.0001)
        assert TestNetworkReduction.ve.query_prob(TestNetworkReduction.network, ["MaryCalls"], Assignment("Burglary")).get_prob(Assignment("MaryCalls")) == pytest.approx(TestNetworkReduction.iz.query_prob(reduced_net3, ["MaryCalls"], Assignment("Burglary")).get_prob(Assignment("MaryCalls")),
                                                                                                                                                                      abs=0.15)
        assert TestNetworkReduction.ve.query_prob(TestNetworkReduction.network, ["MaryCalls"], Assignment("Burglary")).get_prob(Assignment("MaryCalls")) == pytest.approx(TestNetworkReduction.iz.query_prob(reduced_net4, ["MaryCalls"], Assignment("Burglary")).get_prob(Assignment("MaryCalls")),
                                                                                                                                                                      abs=0.15)
        assert TestNetworkReduction.ve.query_prob(TestNetworkReduction.network, ["Earthquake"], Assignment("!MaryCalls")).get_prob(Assignment("Earthquake")) == pytest.approx(
            TestNetworkReduction.ve.query_prob(reduced_net, ["Earthquake"], Assignment("!MaryCalls")).get_prob(Assignment("Earthquake")), abs=0.0001)
        assert TestNetworkReduction.ve.query_prob(TestNetworkReduction.network, ["Earthquake"], Assignment("!MaryCalls")).get_prob(Assignment("Earthquake")) == pytest.approx(
            TestNetworkReduction.ve.query_prob(reduced_net2, ["Earthquake"], Assignment("!MaryCalls")).get_prob(Assignment("Earthquake")), abs=0.0001)
        assert TestNetworkReduction.ve.query_prob(TestNetworkReduction.network, ["Earthquake"], Assignment("!MaryCalls")).get_prob(Assignment("Earthquake")) == pytest.approx(
            TestNetworkReduction.iz.query_prob(reduced_net3, ["Earthquake"], Assignment("!MaryCalls")).get_prob(Assignment("Earthquake")), abs=0.05)
        assert TestNetworkReduction.ve.query_prob(TestNetworkReduction.network, ["Earthquake"], Assignment("!MaryCalls")).get_prob(Assignment("Earthquake")) == pytest.approx(
            TestNetworkReduction.iz.query_prob(reduced_net4, ["Earthquake"], Assignment("!MaryCalls")).get_prob(Assignment("Earthquake")), abs=0.05)

    def test2(self):
        reduced_net = TestNetworkReduction.ve.reduce(TestNetworkReduction.network, ["Burglary", "MaryCalls"], Assignment("!Earthquake"))
        reduced_net2 = TestNetworkReduction.naive.reduce(TestNetworkReduction.network, ["Burglary", "MaryCalls"], Assignment("!Earthquake"))
        reduced_net3 = TestNetworkReduction.iz.reduce(TestNetworkReduction.network, ["Burglary", "MaryCalls"], Assignment("!Earthquake"))
        reduced_net4 = TestNetworkReduction.sw.reduce(TestNetworkReduction.network, ["Burglary", "MaryCalls"], Assignment("!Earthquake"))

        assert len(reduced_net.get_nodes()) == 2
        assert len(reduced_net2.get_nodes()) == 2
        assert len(reduced_net3.get_nodes()) == 2
        assert len(reduced_net4.get_nodes()) == 2
        assert TestNetworkReduction.ve.query_prob(TestNetworkReduction.network, ["MaryCalls"], Assignment("!Earthquake")).get_prob(Assignment("MaryCalls")) == pytest.approx(TestNetworkReduction.ve.query_prob(reduced_net, ["MaryCalls"]).get_prob(Assignment("MaryCalls")), abs=0.0001)
        assert TestNetworkReduction.ve.query_prob(reduced_net, ["MaryCalls"]).get_prob(Assignment("MaryCalls")) == pytest.approx(TestNetworkReduction.naive.query_prob(reduced_net2, ["MaryCalls"]).get_prob(Assignment("MaryCalls")), abs=0.0001)
        assert TestNetworkReduction.ve.query_prob(reduced_net, ["MaryCalls"]).get_prob(Assignment("MaryCalls")) == pytest.approx(TestNetworkReduction.iz.query_prob(reduced_net3, ["MaryCalls"]).get_prob(Assignment("MaryCalls")), abs=0.05)
        assert TestNetworkReduction.ve.query_prob(reduced_net, ["MaryCalls"]).get_prob(Assignment("MaryCalls")) == pytest.approx(TestNetworkReduction.iz.query_prob(reduced_net4, ["MaryCalls"]).get_prob(Assignment("MaryCalls")), abs=0.05)
        assert TestNetworkReduction.ve.query_prob(TestNetworkReduction.network, ["Burglary"], Assignment(["!MaryCalls", "!Earthquake"])).get_prob(Assignment("Burglary")) == pytest.approx(
            TestNetworkReduction.ve.query_prob(reduced_net, ["Burglary"], Assignment("!MaryCalls")).get_prob(Assignment("Burglary")), abs=0.0001)
        assert TestNetworkReduction.ve.query_prob(TestNetworkReduction.network, ["Burglary"], Assignment(["!MaryCalls", "!Earthquake"])).get_prob(Assignment("Burglary")) == pytest.approx(
            TestNetworkReduction.naive.query_prob(reduced_net2, ["Burglary"], Assignment("!MaryCalls")).get_prob(Assignment("Burglary")), abs=0.0001)
        assert TestNetworkReduction.ve.query_prob(reduced_net, ["Burglary"], Assignment("!MaryCalls")).get_prob(Assignment("Burglary")) == pytest.approx(TestNetworkReduction.iz.query_prob(reduced_net3, ["Burglary"], Assignment("!MaryCalls")).get_prob(Assignment("Burglary")),
                                                                                                                                                                     abs=0.05)
        assert TestNetworkReduction.ve.query_prob(reduced_net, ["Burglary"], Assignment("!MaryCalls")).get_prob(Assignment("Burglary")) == pytest.approx(TestNetworkReduction.iz.query_prob(reduced_net4, ["Burglary"], Assignment("!MaryCalls")).get_prob(Assignment("Burglary")),
                                                                                                                                                                     abs=0.05)

    def test3(self):
        reduced_net = TestNetworkReduction.ve.reduce(TestNetworkReduction.network, ["Burglary", "Earthquake"], Assignment("JohnCalls"))
        reduced_net2 = TestNetworkReduction.naive.reduce(TestNetworkReduction.network, ["Burglary", "Earthquake"], Assignment("JohnCalls"))
        reduced_net3 = TestNetworkReduction.iz.reduce(TestNetworkReduction.network, ["Burglary", "Earthquake"], Assignment("JohnCalls"))
        reduced_net4 = TestNetworkReduction.sw.reduce(TestNetworkReduction.network, ["Burglary", "Earthquake"], Assignment("JohnCalls"))

        assert len(reduced_net.get_nodes()) == 2
        assert len(reduced_net2.get_nodes()) == 2
        assert len(reduced_net3.get_nodes()) == 2
        assert len(reduced_net4.get_nodes()) == 2
        assert TestNetworkReduction.ve.query_prob(TestNetworkReduction.network, ["Burglary"], Assignment("JohnCalls")).get_prob(Assignment("Burglary")) == pytest.approx(TestNetworkReduction.iz.query_prob(reduced_net, ["Burglary"]).get_prob(Assignment("Burglary")), abs=0.1)
        assert TestNetworkReduction.ve.query_prob(TestNetworkReduction.network, ["Burglary"], Assignment("JohnCalls")).get_prob(Assignment("Burglary")) == pytest.approx(TestNetworkReduction.naive.query_prob(reduced_net2, ["Burglary"]).get_prob(Assignment("Burglary")), abs=0.0001)
        assert TestNetworkReduction.ve.query_prob(reduced_net, ["Burglary"]).get_prob(Assignment("Burglary")) == pytest.approx(TestNetworkReduction.naive.query_prob(reduced_net3, ["Burglary"]).get_prob(Assignment("Burglary")), abs=0.08)
        assert TestNetworkReduction.ve.query_prob(reduced_net2, ["Burglary"]).get_prob(Assignment("Burglary")) == pytest.approx(TestNetworkReduction.naive.query_prob(reduced_net4, ["Burglary"]).get_prob(Assignment("Burglary")), abs=0.05)
        assert TestNetworkReduction.ve.query_prob(TestNetworkReduction.network, ["Earthquake"], Assignment("JohnCalls")).get_prob(Assignment("Earthquake")) == pytest.approx(TestNetworkReduction.ve.query_prob(reduced_net, ["Earthquake"]).get_prob(Assignment("Earthquake")), abs=0.0001)
        assert TestNetworkReduction.ve.query_prob(reduced_net, ["Earthquake"]).get_prob(Assignment("Earthquake")) == pytest.approx(TestNetworkReduction.iz.query_prob(reduced_net2, ["Earthquake"]).get_prob(Assignment("Earthquake")), abs=0.07)
        assert TestNetworkReduction.ve.query_prob(TestNetworkReduction.network, ["Earthquake"], Assignment("JohnCalls")).get_prob(Assignment("Earthquake")) == pytest.approx(TestNetworkReduction.naive.query_prob(reduced_net3, ["Earthquake"]).get_prob(Assignment("Earthquake")), abs=0.07)
        assert TestNetworkReduction.ve.query_prob(reduced_net, ["Earthquake"]).get_prob(Assignment("Earthquake")) == pytest.approx(TestNetworkReduction.naive.query_prob(reduced_net4, ["Earthquake"]).get_prob(Assignment("Earthquake")), abs=0.07)

    def test5(self):
        reduced_net = TestNetworkReduction.ve.reduce(TestNetworkReduction.network, ["Burglary"], Assignment(["JohnCalls", "MaryCalls"]))

        reduced_net2 = TestNetworkReduction.iz.reduce(TestNetworkReduction.network, ["Burglary"], Assignment(["JohnCalls", "MaryCalls"]))

        reduced_net.add_node(copy(TestNetworkReduction.network.get_node("Action")))
        reduced_net.add_node(copy(TestNetworkReduction.network.get_node("Util1")))
        reduced_net.add_node(copy(TestNetworkReduction.network.get_node("Util2")))
        reduced_net.get_node("Util1").add_input_node(reduced_net.get_node("Burglary"))
        reduced_net.get_node("Util1").add_input_node(reduced_net.get_node("Action"))
        reduced_net.get_node("Util2").add_input_node(reduced_net.get_node("Burglary"))
        reduced_net.get_node("Util2").add_input_node(reduced_net.get_node("Action"))

        table1 = TestNetworkReduction.ve.query_util(reduced_net, "Action")
        table2 = TestNetworkReduction.ve.query_util(TestNetworkReduction.network, ["Action"], Assignment(["JohnCalls", "MaryCalls"]))

        for a in table1.get_table().keys():
            assert table1.get_util(a) == pytest.approx(table2.get_util(a), abs=0.01)

        reduced_net2.add_node(copy(TestNetworkReduction.network.get_node("Action")))
        reduced_net2.add_node(copy(TestNetworkReduction.network.get_node("Util1")))
        reduced_net2.add_node(copy(TestNetworkReduction.network.get_node("Util2")))
        reduced_net2.get_node("Util1").add_input_node(reduced_net2.get_node("Burglary"))
        reduced_net2.get_node("Util1").add_input_node(reduced_net2.get_node("Action"))
        reduced_net2.get_node("Util2").add_input_node(reduced_net2.get_node("Burglary"))
        reduced_net2.get_node("Util2").add_input_node(reduced_net2.get_node("Action"))

        table3 = TestNetworkReduction.ve.query_util(reduced_net2, ["Action"])

        for a in table1.get_table().keys():
            assert table1.get_util(a) == pytest.approx(table3.get_util(a), abs=0.8)

    def test6(self):
        old = copy(TestNetworkReduction.network)

        TestNetworkReduction.network.get_node("Alarm").remove_input_node("Earthquake")
        TestNetworkReduction.network.get_node("Alarm").remove_input_node("Burglary")
        TestNetworkReduction.network.get_chance_node("Alarm").set_distrib(SingleValueDistribution("Alarm", "False"))

        self.test1()
        self.test2()
        self.test3()

        TestNetworkReduction.network = old
