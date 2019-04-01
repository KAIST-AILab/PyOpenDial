from random import Random

from datastructs.assignment import Assignment
from inference.exact.naive_inference import NaiveInference
from test.common.inference_checks import InferenceChecks
from test.common.network_examples import NetworkExamples


class TestInferenceLargeScale:
    percent_comparisions = 0.5

    def test_network(self):
        inference = InferenceChecks()
        inference.include_naive(True)

        bn = NetworkExamples.construct_basic_network2()

        query_vars_powerset = self.generate_powerset(bn.get_chance_node_ids())
        evidence_powerset = self.generate_evidence_powerset(bn)

        nb_errors = 0

        for query_vars in query_vars_powerset:
            if len(query_vars) != 0:
                for evidence in evidence_powerset:
                    if Random().random() < TestInferenceLargeScale.percent_comparisions / 100.0:
                        try:
                            inference.check_prob(bn, query_vars, evidence)
                        except:
                            nb_errors += 1
                            if nb_errors > 2:
                                assert False, "more than 2 sampling errors"

        inference.show_performance()

    def generate_powerset(self, full_set):
        sets = set()

        if len(full_set) == 0:
            sets.add(frozenset(set()))
            return sets

        full_list = list(full_set)
        head = full_list[0]
        rest = set(full_list[1:])

        for s in self.generate_powerset(rest):
            new_set = set()
            new_set.add(head)
            new_set.update(s)
            sets.add(frozenset(new_set))
            sets.add(s)

        return sets

    def generate_evidence_powerset(self, bn):
        all_assignments = []

        full_joint = NaiveInference.get_full_joint(bn, False)

        for a in full_joint.keys():
            partial_assigns = self.generate_powerset(a.get_entry_set())

            for partial in partial_assigns:
                p = Assignment(partial)
                all_assignments.append(p)

        return all_assignments
