import pytest

from bn.distribs.continuous_distribution import ContinuousDistribution
from bn.distribs.density_functions.gaussian_density_function import GaussianDensityFunction
from bn.distribs.density_functions.uniform_density_function import UniformDensityFunction
from bn.distribs.distribution_builder import CategoricalTableBuilder
from bn.distribs.empirical_distribution import EmpiricalDistribution
from bn.distribs.multivariate_table import MultivariateTable
from bn.nodes.chance_node import ChanceNode
from bn.values.value_factory import ValueFactory
from datastructs.assignment import Assignment
from inference.approximate.sampling_algorithm import SamplingAlgorithm
from inference.exact.naive_inference import NaiveInference
from inference.exact.variable_elimination import VariableElimination
from inference.switching_algorithm import SwitchingAlgorithm
from test.common.network_examples import NetworkExamples


class TestInference:
    def test_network1(self):
        bn = NetworkExamples.construct_basic_network()
        full_joint = NaiveInference.get_full_joint(bn, False)

        assert full_joint[Assignment(["JohnCalls", "MaryCalls", "Alarm", "!Burglary", "!Earthquake"])] == pytest.approx(0.000628, abs=0.000001)
        assert full_joint.get(Assignment(["!JohnCalls", "!MaryCalls", "!Alarm", "!Burglary", "!Earthquake"])) == pytest.approx(0.9367428, abs=0.000001)

        naive = NaiveInference()
        query = naive.query_prob(bn, ["Burglary"], Assignment(["JohnCalls", "MaryCalls"]))

        assert query.get_prob(Assignment("Burglary", False)) == pytest.approx(0.71367, abs=0.0001)
        assert query.get_prob(Assignment("Burglary", True)) == pytest.approx(0.286323, abs=0.0001)

        query2 = naive.query_prob(bn, ["Alarm", "Burglary"], Assignment(["Alarm", "MaryCalls"]))

        assert query2.get_prob(Assignment(["Alarm", "!Burglary"])) == pytest.approx(0.623974, abs=0.001)

    def test_network1bis(self):
        bn = NetworkExamples.construct_basic_network2()
        full_joint = NaiveInference.get_full_joint(bn, False)

        assert full_joint.get(Assignment(["JohnCalls", "MaryCalls", "Alarm", "!Burglary", "!Earthquake"])) == pytest.approx(0.000453599, abs=0.000001)
        assert full_joint.get(Assignment(["!JohnCalls", "!MaryCalls", "!Alarm", "!Burglary", "!Earthquake"])) == pytest.approx(0.6764828, abs=0.000001)

        naive = NaiveInference()
        query = naive.query_prob(bn, ["Burglary"], Assignment(["JohnCalls", "MaryCalls"]))

        assert query.get_prob(Assignment("Burglary", False)) == pytest.approx(0.360657, abs=0.0001)
        assert query.get_prob(Assignment("Burglary", True)) == pytest.approx(0.639343, abs=0.0001)

        query2 = naive.query_prob(bn, ["Alarm", "Burglary"], Assignment(["Alarm", "MaryCalls"]))

        assert query2.get_prob(Assignment(["Alarm", "!Burglary"])) == pytest.approx(0.3577609, abs=0.001)

    def test_network2(self):
        ve = VariableElimination()
        bn = NetworkExamples.construct_basic_network()

        distrib = ve.query_prob(bn, ["Burglary"], Assignment(["JohnCalls", "MaryCalls"]))

        assert distrib.get_prob(Assignment("Burglary", False)) == pytest.approx(0.713676, abs=0.0001)
        assert distrib.get_prob(Assignment("Burglary", True)) == pytest.approx(0.286323, abs=0.0001)

        query2 = ve.query_prob(bn, ["Alarm", "Burglary"], Assignment(["Alarm", "MaryCalls"]))

        assert query2.get_prob(Assignment(["Alarm", "!Burglary"])) == pytest.approx(0.623974, abs=0.001)

    def test_network2bis(self):
        ve = VariableElimination()
        bn = NetworkExamples.construct_basic_network2()

        query = ve.query_prob(bn, ["Burglary"], Assignment(["JohnCalls", "MaryCalls"]))

        assert query.get_prob(Assignment("Burglary", False)) == pytest.approx(0.360657, abs=0.0001)
        assert query.get_prob(Assignment("Burglary", True)) == pytest.approx(0.63934, abs=0.0001)

        query2 = ve.query_prob(bn, ["Alarm", "Burglary"], Assignment(["Alarm", "MaryCalls"]))

        assert query2.get_prob(Assignment(["Alarm", "!Burglary"])) == pytest.approx(0.3577609, abs=0.001)

    def test_network3bis(self):
        iz = SamplingAlgorithm(5000, 300)
        bn = NetworkExamples.construct_basic_network2()

        query = iz.query_prob(bn, ["Burglary"], Assignment(["JohnCalls", "MaryCalls"]))

        assert query.get_prob(Assignment("Burglary", False)) == pytest.approx(0.362607, abs=0.06)
        assert query.get_prob(Assignment("Burglary", True)) == pytest.approx(0.637392, abs=0.06)

        query2 = iz.query_prob(bn, ["Alarm", "Burglary"], Assignment(["Alarm", "MaryCalls"]))

        assert query2.get_prob(Assignment(["Alarm", "!Burglary"])) == pytest.approx(0.35970, abs=0.05)

    def test_network_util(self):
        network = NetworkExamples.construct_basic_network2()
        ve = VariableElimination()
        naive = NaiveInference()
        iz = SamplingAlgorithm(4000, 300)

        assert ve.query_util(network, ["Action"], Assignment([Assignment("JohnCalls"), Assignment("MaryCalls")])).get_util(Assignment("Action", "CallPolice")) == pytest.approx(-0.680, abs=0.001)
        assert naive.query_util(network, ["Action"], Assignment([Assignment("JohnCalls"), Assignment("MaryCalls")])).get_util(Assignment("Action", "CallPolice")) == pytest.approx(-0.680, abs=0.001)
        assert iz.query_util(network, ["Action"], Assignment([Assignment("JohnCalls"), Assignment("MaryCalls")])).get_util(Assignment("Action", "CallPolice")) == pytest.approx(-0.680, abs=0.5)
        assert ve.query_util(network, ["Action"], Assignment([Assignment("JohnCalls"), Assignment("MaryCalls")])).get_util(Assignment("Action", "DoNothing")) == pytest.approx(-6.213, abs=0.001)
        assert naive.query_util(network, ["Action"], Assignment([Assignment("JohnCalls"), Assignment("MaryCalls")])).get_util(Assignment("Action", "DoNothing")) == pytest.approx(-6.213, abs=0.001)
        assert iz.query_util(network, ["Action"], Assignment([Assignment("JohnCalls"), Assignment("MaryCalls")])).get_util(Assignment("Action", "DoNothing")) == pytest.approx(-6.213, abs=1.5)
        assert ve.query_util(network, ["Burglary"], Assignment([Assignment("JohnCalls"), Assignment("MaryCalls")])).get_util(Assignment("!Burglary")) == pytest.approx(-0.1667, abs=0.001)
        assert naive.query_util(network, ["Burglary"], Assignment([Assignment("JohnCalls"), Assignment("MaryCalls")])).get_util(Assignment("!Burglary")) == pytest.approx(-0.1667, abs=0.001)
        assert iz.query_util(network, ["Burglary"], Assignment([Assignment("JohnCalls"), Assignment("MaryCalls")])).get_util(Assignment("!Burglary")) == pytest.approx(-0.25, abs=0.5)
        assert ve.query_util(network, ["Burglary"], Assignment([Assignment("JohnCalls"), Assignment("MaryCalls")])).get_util(Assignment("Burglary")) == pytest.approx(-3.5, abs=0.001)
        assert naive.query_util(network, ["Burglary"], Assignment([Assignment("JohnCalls"), Assignment("MaryCalls")])).get_util(Assignment("Burglary")) == pytest.approx(-3.5, abs=0.001)
        assert iz.query_util(network, ["Burglary"], Assignment([Assignment("JohnCalls"), Assignment("MaryCalls")])).get_util(Assignment("Burglary")) == pytest.approx(-3.5, abs=1.0)

    def test_switching(self):
        old_factor = SwitchingAlgorithm.max_branching_factor
        SwitchingAlgorithm.max_branching_factor = 4
        network = NetworkExamples.construct_basic_network2()
        distrib = SwitchingAlgorithm().query_prob(network, ["Burglary"], Assignment(["JohnCalls", "MaryCalls"]))

        assert isinstance(distrib, MultivariateTable)

        builder = CategoricalTableBuilder("n1")
        builder.add_row(ValueFactory.create("aha"), 1.0)

        n1 = ChanceNode("n1", builder.build())
        network.add_node(n1)
        builder = CategoricalTableBuilder("n2")
        builder.add_row(ValueFactory.create("oho"), 0.7)

        n2 = ChanceNode("n2", builder.build())
        network.add_node(n2)
        builder = CategoricalTableBuilder("n3")
        builder.add_row(ValueFactory.create("ihi"), 0.7)

        n3 = ChanceNode("n3", builder.build())
        network.add_node(n3)
        network.get_node("Alarm").add_input_node(n1)
        network.get_node("Alarm").add_input_node(n2)
        network.get_node("Alarm").add_input_node(n3)

        distrib = SwitchingAlgorithm().query_prob(network, ["Burglary"], Assignment(["JohnCalls", "MaryCalls"]))
        assert distrib.__class__ == EmpiricalDistribution

        network.remove_node(n1.get_id())
        network.remove_node(n2.get_id())

        distrib = SwitchingAlgorithm().query_prob(network, ["Burglary"], Assignment(["JohnCalls", "MaryCalls"]))
        assert isinstance(distrib, MultivariateTable)

        n1 = ChanceNode("n1", ContinuousDistribution("n1", UniformDensityFunction(-2.0, 2.0)))
        n2 = ChanceNode("n2", ContinuousDistribution("n2", GaussianDensityFunction(-1.0, 3.0)))

        network.add_node(n1)
        network.add_node(n2)
        network.get_node("Earthquake").add_input_node(n1)
        network.get_node("Earthquake").add_input_node(n2)

        distrib = SwitchingAlgorithm().query_prob(network, ["Burglary"], Assignment(["JohnCalls", "MaryCalls"]))
        assert isinstance(distrib, EmpiricalDistribution)

        SwitchingAlgorithm.max_branching_factor = old_factor
