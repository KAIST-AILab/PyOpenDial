import math

import numpy as np
import pytest

from bn.b_network import BNetwork
from bn.distribs.conditional_table import ConditionalTable
from bn.distribs.continuous_distribution import ContinuousDistribution
from bn.distribs.density_functions.dirichlet_density_function import DirichletDensityFunction
from bn.distribs.density_functions.gaussian_density_function import GaussianDensityFunction
from bn.distribs.density_functions.kernel_density_function import KernelDensityFunction
from bn.distribs.density_functions.uniform_density_function import UniformDensityFunction
from bn.distribs.distribution_builder import CategoricalTableBuilder as CategoricalTableBuilder, \
    ConditionalTableBuilder as ConditionalTableBuilder, MultivariateTableBuilder as MultivariateTableBuilder
from bn.nodes.chance_node import ChanceNode
from bn.values.array_val import ArrayVal
from bn.values.value_factory import ValueFactory
from datastructs.assignment import Assignment
from inference.approximate.sampling_algorithm import SamplingAlgorithm
from inference.exact.variable_elimination import VariableElimination
from settings import Settings
from test.common.inference_checks import InferenceChecks
from utils.math_utils import MathUtils


class TestDistribution:
    def test_simple_distrib(self):
        builder = CategoricalTableBuilder("var1")
        builder.add_row("val1", 0.7)
        assert not builder.is_well_formed()
        builder.add_row("val2", 0.3)
        assert builder.is_well_formed()
        table = builder.build()

        assert table.get_prob("val1") == pytest.approx(0.7, abs=0.001)
        assert table.get_prob("val1") == pytest.approx(0.7, abs=0.001)
        table2 = MultivariateTableBuilder()
        table2.add_row(Assignment(Assignment("var2", "val3"), "var1", "val2"), 0.9)
        # assert not table2.is_well_formed()
        table2.add_row(Assignment(Assignment("var2", "val3"), "var1", "val1"), 0.1)
        assert table2.is_well_formed()
        assert table2.build().get_prob(Assignment([Assignment("var1", "val1"), Assignment("var2", "val3")])) == pytest.approx(0.1, abs=0.001)

    def test_maths(self):
        assert MathUtils.get_volume(2.0, 1) == pytest.approx(4.0, abs=0.001)
        assert MathUtils.get_volume(2.0, 2) == pytest.approx(math.pi * 4, abs=0.001)
        assert MathUtils.get_volume(2.0, 3) == pytest.approx(4.0 / 3.0 * math.pi * 8, abs=0.001)
        assert MathUtils.get_volume(3.0, 4) == pytest.approx(math.pow(math.pi, 2) / 2 * 81, abs=0.001)

    def test_conversion1_distrib(self):
        builder = CategoricalTableBuilder("var1")

        builder.add_row(1.5, 0.7)
        builder.add_row(2.0, 0.1)
        builder.add_row(-3.0, 0.2)
        assert builder.is_well_formed()
        table = builder.build()
        assert table.get_prob("2.0") == pytest.approx(0.1, abs=0.001)

        continuous = table.to_continuous()
        assert continuous.get_prob_density(2.0) == pytest.approx(0.2, abs=0.001)
        assert continuous.get_prob_density(2.1) == pytest.approx(0.2, abs=0.001)
        assert continuous.get_cumulative_prob(-3.1) == pytest.approx(0.0, abs=0.001)
        assert continuous.get_cumulative_prob(1.6) == pytest.approx(0.9, abs=0.001)

        table2 = continuous.to_discrete()
        assert len(table2.get_values()) == 3
        assert table2.get_prob(2.0) == pytest.approx(0.1, abs=0.05)

        sum = 0
        for _ in range(10000):
            sum += continuous.sample().get_double()

        assert sum / 10000.0 == pytest.approx(0.65, abs=0.1)

    def test_continuous(self):
        distrib = ContinuousDistribution("X", UniformDensityFunction(-2.0, 4.0))
        assert distrib.get_prob_density(1.0) == pytest.approx(1 / 6.0, abs=0.0001)
        assert distrib.get_prob_density(4.0) == pytest.approx(1 / 6.0, abs=0.0001)
        assert distrib.get_prob_density(-3.0) == pytest.approx(0.0, abs=0.0001)
        assert distrib.get_prob_density(6.0) == pytest.approx(0.0, abs=0.0001)
        assert distrib.to_discrete().get_prob(0.5) == pytest.approx(0.01, abs=0.01)
        assert distrib.to_discrete().get_prob(4) == pytest.approx(0.01, abs=0.01)

        total_prob = 0.0
        for value in distrib.to_discrete().get_posterior(Assignment()).get_values():
            total_prob += distrib.to_discrete().get_prob(value)

        assert total_prob == pytest.approx(1.0, abs=0.03)

    def test_gaussian(self):
        distrib = ContinuousDistribution("X", GaussianDensityFunction(1.0, 3.0))
        assert distrib.get_prob_density(1.0) == pytest.approx(0.23032, abs=0.001)
        assert distrib.get_prob_density(-3.0) == pytest.approx(0.016, abs=0.01)
        assert distrib.get_prob_density(6.0) == pytest.approx(0.00357, abs=0.01)
        assert distrib.to_discrete().get_prob(1.0) == pytest.approx(0.06290, abs=0.01)
        assert distrib.to_discrete().get_prob(0.5) == pytest.approx(0.060615, abs=0.01)
        assert distrib.to_discrete().get_prob(4) == pytest.approx(0.014486, abs=0.01)

        total_prob = 0.0
        for value in distrib.to_discrete().get_posterior(Assignment()).get_values():
            total_prob += distrib.to_discrete().get_prob(value)

        assert total_prob == pytest.approx(1.0, abs=0.05)
        assert distrib.get_function().get_mean()[0] == pytest.approx(1.0, abs=0.01)
        assert distrib.get_function().get_variance()[0] == pytest.approx(3.0, abs=0.01)

        samples = list()
        for _ in range(20000):
            val = [distrib.sample().get_double()]
            samples.append(val)
        samples = np.array(samples)

        estimated = GaussianDensityFunction(samples)
        assert estimated.get_mean()[0] == pytest.approx(distrib.get_function().get_mean()[0], abs=0.05)
        assert estimated.get_variance()[0] == pytest.approx(distrib.get_function().get_variance()[0], abs=0.1)

    def test_discrete(self):
        builder = CategoricalTableBuilder("A")

        builder.add_row(1.0, 0.6)
        builder.add_row(2.5, 0.3)
        table = builder.build()
        assert table.get_prob(2.5) == pytest.approx(0.3, abs=0.0001)
        assert table.get_prob(1.0) == pytest.approx(0.6, abs=0.0001)

        distrib = table.to_continuous()
        assert distrib.get_prob_density(2.5) == pytest.approx(0.2, abs=0.01)
        assert distrib.get_prob_density(1.0) == pytest.approx(0.4, abs=0.001)
        assert distrib.get_prob_density(-2.0) == pytest.approx(0, abs=0.001)
        assert distrib.get_prob_density(0.9) == pytest.approx(0.4, abs=0.001)
        assert distrib.get_prob_density(1.2) == pytest.approx(0.4, abs=0.0001)
        assert distrib.get_prob_density(2.2) == pytest.approx(0.2, abs=0.001)
        assert distrib.get_prob_density(2.7) == pytest.approx(0.2, abs=0.001)
        assert distrib.get_prob_density(5.0) == pytest.approx(0, abs=0.0001)
        assert distrib.get_cumulative_prob(0.5) == pytest.approx(0, abs=0.0001)
        assert distrib.get_cumulative_prob(1.1) == pytest.approx(0.6, abs=0.0001)
        assert distrib.get_cumulative_prob(2.4) == pytest.approx(0.6, abs=0.0001)
        assert distrib.get_cumulative_prob(2.5) == pytest.approx(0.9, abs=0.0001)
        assert distrib.get_cumulative_prob(2.6) == pytest.approx(0.9, abs=0.0001)

        assert distrib.get_function().get_mean()[0] == pytest.approx(1.35, abs=0.01)
        assert distrib.get_function().get_variance()[0] == pytest.approx(0.47, abs=0.01)

    def test_uniform_distrib(self):
        continuous2 = ContinuousDistribution("var2", UniformDensityFunction(-2.0, 3.0))
        assert continuous2.get_prob_density(1.2) == pytest.approx(1 / 5.0, abs=0.001)
        # assert continuous2.get_cumulative_prob(Assignment("var2", # 2)), 4 / 5.0 == 0.001
        assert len(continuous2.to_discrete().get_values()) == Settings.discretization_buckets
        assert continuous2.get_prob(ValueFactory.create(1.2)) == pytest.approx(0.01, abs=0.01)

        sum = 0.
        for _ in range(10000):
            sum += continuous2.sample().get_double()

        assert sum / 10000.0 == pytest.approx(0.5, abs=0.1)
        assert continuous2.get_function().get_mean()[0] == pytest.approx(0.5, abs=0.01)
        assert continuous2.get_function().get_variance()[0] == pytest.approx(2.08, abs=0.01)

    def test_gaussian_distrib(self):
        continuous2 = ContinuousDistribution("var2", GaussianDensityFunction(2.0, 3.0))
        assert continuous2.get_prob_density(1.2) == pytest.approx(0.2070, abs=0.001)
        assert continuous2.get_prob_density(2.0) == pytest.approx(0.23033, abs=0.001)
        assert continuous2.get_cumulative_prob(2.0) == pytest.approx(0.5, abs=0.001)
        assert continuous2.get_cumulative_prob(3.0) == pytest.approx(0.7181, abs=0.001)
        assert len(continuous2.to_discrete().get_values()) > Settings.discretization_buckets / 2
        assert len(continuous2.to_discrete().get_values()) <= Settings.discretization_buckets
        assert continuous2.to_discrete().get_prob(2) == pytest.approx(0.06205, abs=0.01)

        sum = 0.
        for _ in range(10000):
            sum += continuous2.sample().get_double()

        assert sum / 10000.0 == pytest.approx(2.0, abs=0.1)

    def test_kernel_distrib(self):
        kds = KernelDensityFunction(np.array([[0.1], [- 1.5], [0.6], [1.3], [1.3]]))

        continuous2 = ContinuousDistribution("var2", kds)
        assert continuous2.get_prob_density(-2.0) == pytest.approx(0.086, abs=0.01)
        assert continuous2.get_prob_density(0.6) == pytest.approx(0.32, abs=0.1)
        assert continuous2.get_prob_density(1.3) == pytest.approx(0.30, abs=0.1)
        assert continuous2.get_cumulative_prob(ValueFactory.create(-1.6)) == pytest.approx(0.0, abs=0.01)
        assert continuous2.get_cumulative_prob(ValueFactory.create(-1.4)) == pytest.approx(0.2, abs=0.01)
        assert continuous2.get_cumulative_prob(ValueFactory.create(1.29)) == pytest.approx(0.6, abs=0.01)
        assert continuous2.get_cumulative_prob(ValueFactory.create(1.3)) == pytest.approx(1.0, abs=0.01)
        assert continuous2.get_cumulative_prob(ValueFactory.create(1.31)) == pytest.approx(1.0, abs=0.01)

        sum = 0.
        for _ in range(20000):
            sum += continuous2.sample().get_double()

        assert sum / 20000.0 == pytest.approx(0.424, abs=0.1)
        # DistributionViewer.showDistributionViewer(continuous2)
        # Thread.sleep(300000000)
        assert continuous2.to_discrete().get_prob(-1.5) == pytest.approx(0.2, abs=0.05)

        assert continuous2.get_function().get_mean()[0] == pytest.approx(0.36, abs=0.01)
        assert continuous2.get_function().get_variance()[0] == pytest.approx(1.07, abs=0.01)

    def test_empirical_distrib(self):
        st = CategoricalTableBuilder("var1")

        st.add_row("val1", 0.6)
        st.add_row("val2", 0.4)

        builder = ConditionalTableBuilder("var2")
        builder.add_row(Assignment("var1", "val1"), "val1", 0.9)
        builder.add_row(Assignment("var1", "val1"), "val2", 0.1)
        builder.add_row(Assignment("var1", "val2"), "val1", 0.2)
        builder.add_row(Assignment("var1", "val2"), "val2", 0.8)

        bn = BNetwork()
        var1 = ChanceNode("var1", st.build())
        bn.add_node(var1)

        var2 = ChanceNode("var2", builder.build())
        var2.add_input_node(var1)
        bn.add_node(var2)

        sampling = SamplingAlgorithm(2000, 500)

        distrib = sampling.query_prob(bn, "var2", Assignment("var1", "val1"))
        assert distrib.get_prob("val1") == pytest.approx(0.9, abs=0.05)
        assert distrib.get_prob("val2") == pytest.approx(0.1, abs=0.05)

        distrib2 = sampling.query_prob(bn, "var2")
        assert distrib2.get_prob("val1") == pytest.approx(0.62, abs=0.05)
        assert distrib2.get_prob("val2") == pytest.approx(0.38, abs=0.05)

    def test_empirical_distrib_continuous(self):
        continuous = ContinuousDistribution("var1", UniformDensityFunction(-1.0, 3.0))

        bn = BNetwork()
        var1 = ChanceNode("var1", continuous)
        bn.add_node(var1)

        sampling = SamplingAlgorithm(2000, 200)

        distrib2 = sampling.query_prob(bn, "var1")
        assert len(distrib2.get_posterior(Assignment()).get_values()) == pytest.approx(Settings.discretization_buckets, abs=2)
        assert distrib2.to_continuous().get_cumulative_prob(-1.1) == pytest.approx(0, abs=0.001)
        assert distrib2.to_continuous().get_cumulative_prob(1.0) == pytest.approx(0.5, abs=0.06)
        assert distrib2.to_continuous().get_cumulative_prob(3.1) == pytest.approx(1.0, abs=0.00)

        assert continuous.get_prob_density(-2.0) == pytest.approx(distrib2.to_continuous().get_prob_density(-2.0), abs=0.1)
        assert continuous.get_prob_density(-0.5) == pytest.approx(distrib2.to_continuous().get_prob_density(-0.5), abs=0.1)
        assert continuous.get_prob_density(1.8) == pytest.approx(distrib2.to_continuous().get_prob_density(1.8), abs=0.1)
        assert continuous.get_prob_density(3.2) == pytest.approx(distrib2.to_continuous().get_prob_density(3.2), abs=0.1)

    def test_dep_empirical_distrib_continuous(self):
        bn = BNetwork()
        builder = CategoricalTableBuilder("var1")
        builder.add_row(ValueFactory.create("one"), 0.7)
        builder.add_row(ValueFactory.create("two"), 0.3)
        var1 = ChanceNode("var1", builder.build())
        bn.add_node(var1)

        continuous = ContinuousDistribution("var2", UniformDensityFunction(-1.0, 3.0))
        continuous2 = ContinuousDistribution("var2", GaussianDensityFunction(3.0, 10.0))

        table = ConditionalTable("var2")
        table.add_distrib(Assignment("var1", "one"), continuous)
        table.add_distrib(Assignment("var1", "two"), continuous2)
        var2 = ChanceNode("var2", table)
        var2.add_input_node(var1)
        bn.add_node(var2)

        inference = InferenceChecks()
        inference.check_cdf(bn, "var2", -1.5, 0.021)
        inference.check_cdf(bn, "var2", 0., 0.22)
        inference.check_cdf(bn, "var2", 2., 0.632)
        inference.check_cdf(bn, "var2", 8., 0.98)

        #
        # ProbDistribution
        # distrib = (ImportanceSampling()).query_prob(query)
        # DistributionViewer.showDistributionViewer(distrib)
        # Thread.sleep(300000000)
        # /

    def test_dirichlet(self):
        old_discretisation_settings = Settings.discretization_buckets
        Settings.discretization_buckets = 250

        alphas = list()
        alphas.append(40.0)
        alphas.append(80.0)
        alphas = np.array(alphas)

        dirichlet = DirichletDensityFunction(alphas)
        distrib = ContinuousDistribution("x", dirichlet)
        assert isinstance(distrib.sample(), ArrayVal)

        assert 2 == len(distrib.sample())
        assert distrib.sample().get_array()[0] == pytest.approx(0.33, abs=0.15)

        ##############################################
        # dirichlet distribution 자바 코드에 버그가 있음.
        ##############################################
        # assert distrib.get_prob_density(ArrayVal([1./3, 2./3])) == pytest.approx(8.0, abs=0.5)

        n = ChanceNode("x", distrib)
        network = BNetwork()
        network.add_node(n)

        table = VariableElimination().query_prob(network, "x")

        sum = 0.
        for value in table.get_values():
            if value.get_array()[0] < 0.33333:
                sum += table.get_prob(value)

        assert sum == pytest.approx(0.5, abs=0.1)

        conversion1 = VariableElimination().query_prob(network, "x")

        assert abs(len(conversion1.get_posterior(Assignment()).get_values()) - Settings.discretization_buckets) < 10
        assert conversion1.get_posterior(Assignment()).get_prob(ValueFactory.create("[0.3333,0.6666]")) == pytest.approx(0.02, abs=0.05)

        conversion3 = SamplingAlgorithm(4000, 1000).query_prob(network, "x")

        # DistributionViewer(conversion3)
        # Thread.sleep(3000000)

        # TODO: 아래 테스트 케이스 문제 없는지 확인 필요.
        # assert conversion3.to_continuous().get_prob_density(ValueFactory.create("[0.3333,0.6666]")) == pytest.approx(9.0, abs=1.5)

        assert distrib.get_function().get_mean()[0] == pytest.approx(0.333333, abs=0.01)
        assert distrib.get_function().get_variance()[0] == pytest.approx(0.002, abs=0.01)

        assert conversion3.to_continuous().get_function().get_mean()[0] == pytest.approx(0.333333, abs=0.05)
        assert conversion3.to_continuous().get_function().get_variance()[0] == pytest.approx(0.002, abs=0.05)

        Settings.discretization_buckets = old_discretisation_settings

    def test_dirichlet2(self):
        alpha_list = np.array([2, 3])
        dirichlet = DirichletDensityFunction(alpha_list)
        assert dirichlet.get_density(np.array([0.3, 0.7])) == pytest.approx(1.763999, abs=0.01)

    def test_kernel_distrib2(self):
        mkds = KernelDensityFunction(np.array([[0.1], [-1.5], [0.6], [1.3], [1.3]]))

        continuous2 = ContinuousDistribution("var2", mkds)

        assert continuous2.get_prob_density(np.array([-2.0])) == pytest.approx(0.086, abs=0.003)
        assert continuous2.get_prob_density(np.array([0.6])) == pytest.approx(0.32, abs=0.02)
        assert continuous2.get_prob_density(np.array([1.3])) == pytest.approx(0.30, abs=0.02)

        sum = 0.
        for _ in range(10000):
            sum += continuous2.sample().get_double()

        assert sum / 10000.0 == pytest.approx(0.424, abs=0.15)
        assert continuous2.to_discrete().get_prob(-1.5) == pytest.approx(0.2, abs=0.1)

    def test_nbest(self):
        builder = CategoricalTableBuilder("test")

        builder.add_row("bla", 0.5)
        builder.add_row("blo", 0.1)
        table = builder.build()
        for _ in range(10):
            assert str(table.get_best()) == "bla"

        builder2 = MultivariateTableBuilder()
        builder2.add_row(Assignment("test", "bla"), 0.5)
        builder2.add_row(Assignment("test", "blo"), 0.1)

        table2 = builder2.build()
        for _ in range(10):
            assert str(table2.get_best().get_value("test")) == "bla"
