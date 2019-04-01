import logging
from collections import Collection
from math import isclose

from multipledispatch import dispatch

from bn.b_network import BNetwork
from bn.values.value import Value
from datastructs.assignment import Assignment
from inference.approximate.sampling_algorithm import SamplingAlgorithm
from inference.exact.naive_inference import NaiveInference
from inference.exact.variable_elimination import VariableElimination
from inference.query import ProbQuery, UtilQuery
from utils.py_utils import current_time_millis


class InferenceChecks:
    exact_threshold = 0.01
    sampling_threshold = 0.1

    # logger
    log = logging.getLogger('PyOpenDial')

    def __init__(self):
        self._variable_elimination = VariableElimination()
        self._sampling_algorithm_1 = SamplingAlgorithm(2000, 200)
        self._sampling_algorithm_2 = SamplingAlgorithm(15000, 1500)
        self._naive_inference = NaiveInference()

        self._timings = dict()
        self._numbers = dict()

        self._timings[self._variable_elimination] = 0
        self._numbers[self._variable_elimination] = 0
        self._timings[self._sampling_algorithm_1] = 0
        self._numbers[self._sampling_algorithm_1] = 0
        self._timings[self._sampling_algorithm_2] = 0
        self._numbers[self._sampling_algorithm_2] = 0
        self._timings[self._naive_inference] = 0
        self._numbers[self._naive_inference] = 0

        self._include_naive = False

    def include_naive(self, include_naive):
        self._include_naive = include_naive

    @dispatch(BNetwork, Collection, Assignment)
    def check_prob(self, network, query_vars, evidence):
        query = ProbQuery(network, query_vars, Assignment())
        distrib_1 = self._compute_prob(query, self._variable_elimination)
        distrib_2 = self._compute_prob(query, self._sampling_algorithm_1)
        try:
            self._compare_distributions(distrib_1, distrib_2, 0.1)
        except AssertionError as e:
            distrib_2 = self._compute_prob(query, self._sampling_algorithm_2)
            self.log.debug('resampling for query %s' % (str(ProbQuery(network, query_vars, evidence))))
            self._compare_distributions(distrib_1, distrib_2, 0.1)

        if self._include_naive:
            distrib_3 = self._compute_prob(query, self._naive_inference)
            self._compare_distributions(distrib_1, distrib_3, 0.01)

    @dispatch(BNetwork, str, str, float)
    def check_prob(self, network, query_var, a, expected):
        self.check_prob(network, [query_var], Assignment(query_var, a), expected)

    @dispatch(BNetwork, str, Value, float)
    def check_prob(self, network, query_var, a, expected):
        self.check_prob(network, [query_var], Assignment(query_var, a), expected)

    @dispatch(BNetwork, Collection, Assignment, float)
    def check_prob(self, network, query_vars, a, expected):
        query = ProbQuery(network, query_vars, Assignment())
        distrib_1 = self._compute_prob(query, self._variable_elimination)
        distrib_2 = self._compute_prob(query, self._sampling_algorithm_1)

        assert isclose(expected, distrib_1.get_prob(a), abs_tol=InferenceChecks.exact_threshold)
        try:
            assert isclose(expected, distrib_2.get_prob(a), abs_tol=InferenceChecks.sampling_threshold)
        except AssertionError as e:
            distrib_2 = self._compute_prob(query, self._sampling_algorithm_2)
            assert isclose(expected, distrib_2.get_prob(a), abs_tol=InferenceChecks.sampling_threshold)

        if self._include_naive:
            distrib_3 = self._compute_prob(query, self._naive_inference)
            assert isclose(expected, distrib_3.get_prob(a), abs_tol=InferenceChecks.exact_threshold)

    @dispatch(BNetwork, str, float, float)
    def check_cdf(self, network, variable, value, expected):
        query = ProbQuery(network, [variable], Assignment())
        distrib_1 = self._compute_prob(query, self._variable_elimination).get_marginal(variable).to_continuous()
        distrib_2 = self._compute_prob(query, self._sampling_algorithm_1).get_marginal(variable).to_continuous()

        assert isclose(expected, distrib_1.get_cumulative_prob(value), abs_tol=InferenceChecks.exact_threshold)

        try:
            assert isclose(expected, distrib_2.get_cumulative_prob(value), abs_tol=InferenceChecks.sampling_threshold)
        except AssertionError as e:
            distrib_2 = self._compute_prob(query, self._sampling_algorithm_2).get_marginal(variable).to_continuous()
            assert isclose(expected, distrib_2.get_cumulative_prob(value), abs_tol=InferenceChecks.sampling_threshold)

        if self._include_naive:
            distrib_3 = self._compute_prob(query, self._naive_inference).get_marginal(variable).to_continuous()
            assert isclose(expected, distrib_3.to_discrete().get_prob(value), abs_tol=InferenceChecks.exact_threshold)

    @dispatch(BNetwork, str, str, float)
    def check_util(self, network, query_var, a, expected):
        self.check_util(network, [query_var], Assignment(query_var, a), expected)

    @dispatch(BNetwork, Collection, Assignment, float)
    def check_util(self, network, query_vars, a, expected):
        query = UtilQuery(network, query_vars, Assignment())
        distrib_1 = self._compute_util(query, self._variable_elimination)
        distrib_2 = self._compute_util(query, self._sampling_algorithm_1)

        assert isclose(expected, distrib_1.get_util(a), abs_tol=InferenceChecks.exact_threshold)

        try:
            assert isclose(expected, distrib_2.get_util(a), abs_tol=InferenceChecks.sampling_threshold * 5)
        except AssertionError as e:
            distrib_2 = self._compute_util(query, self._sampling_algorithm_2)
            assert isclose(expected, distrib_2.get_util(a), abs_tol=InferenceChecks.sampling_threshold * 5)

        if self._include_naive:
            distrib_3 = self._compute_util(query, self._naive_inference)
            assert isclose(expected, distrib_3.get_util(a), abs_tol=InferenceChecks.exact_threshold)

    def _compute_prob(self, query, inference_algorithm):
        start_time = current_time_millis()
        distrib = inference_algorithm.query_prob(query)
        inference_time = current_time_millis() - start_time
        self._numbers[inference_algorithm] = self._numbers[inference_algorithm] + 1
        self._timings[inference_algorithm] = self._timings[inference_algorithm] + inference_time

        return distrib

    def _compute_util(self, query, inference_algorithm):
        start_time = current_time_millis()
        distrib = inference_algorithm.query_util(query)
        inference_time = current_time_millis() - start_time

        self._numbers[inference_algorithm] = self._numbers[inference_algorithm] + 1
        self._timings[inference_algorithm] = self._timings[inference_algorithm] + inference_time

        return distrib

    def _compare_distributions(self, distrib_1, distrib_2, margin):
        rows = distrib_1.get_values()
        for row in rows:
            assert isclose(distrib_1.get_prob(row), distrib_2.get_prob(row), abs_tol=margin)

    def show_performance(self):
        if self._include_naive:
            print('Average time for naive inference: %f (ms)' % (self._timings[self._naive_inference] / 1000000.0 / self._numbers[self._naive_inference]))

        print('Average time for variable elimination: %f (ms)' % (self._timings[self._variable_elimination] / 1000000.0 / self._numbers[self._variable_elimination]))

        importance_sampling_time = (self._timings[self._sampling_algorithm_1] + self._timings[self._sampling_algorithm_2]) / 1000000.0 / self._numbers[self._sampling_algorithm_1]
        repeats = self._numbers[self._sampling_algorithm_2] * 100. / self._numbers[self._sampling_algorithm_1]
        print('Average time for importance sampling: %f (ms) with %f %% of repeats.' % (importance_sampling_time, repeats))
