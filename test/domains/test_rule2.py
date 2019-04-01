from bn.distribs.distribution_builder import CategoricalTableBuilder
from bn.values.value_factory import ValueFactory
from datastructs.assignment import Assignment
from dialogue_system import DialogueSystem
from domains.rules.distribs.equivalence_distribution import EquivalenceDistribution
from modules.forward_planner import ForwardPlanner
from modules.state_pruner import StatePruner
from readers.xml_domain_reader import XMLDomainReader
from test.common.inference_checks import InferenceChecks


class TestRule2:
    domain_file = "test/data/domain2.xml"
    domain_file2 = "test/data/domain3.xml"
    domain_file3 = "test/data/domain4.xml"
    domain_file4 = "test/data/thesistest2.xml"

    domain = XMLDomainReader.extract_domain(domain_file)
    inference = InferenceChecks()

    def test_1(self):
        system = DialogueSystem(TestRule2.domain)
        eq_factor = EquivalenceDistribution.none_prob
        EquivalenceDistribution.none_prob = 0.1
        old_prune_threshold = StatePruner.value_pruning_threshold
        StatePruner.value_pruning_threshold = 0.0

        system.get_settings().show_gui = False
        system.detach_module(ForwardPlanner)
        system.start_system()

        TestRule2.inference.check_prob(system.get_state(), "a_u^p", "Ask(A)", 0.63)
        TestRule2.inference.check_prob(system.get_state(), "a_u^p", "Ask(B)", 0.27)
        TestRule2.inference.check_prob(system.get_state(), "a_u^p", "None", 0.1)

        builder = CategoricalTableBuilder("a_u")
        builder.add_row("Ask(B)", 0.8)
        builder.add_row("None", 0.2)

        system.get_state().remove_nodes(system.get_state().get_action_node_ids())
        system.get_state().remove_nodes(system.get_state().get_utility_node_ids())

        system.add_content(builder.build())

        TestRule2.inference.check_prob(system.get_state(), "i_u", "Want(A)", 0.090)
        TestRule2.inference.check_prob(system.get_state(), "i_u", "Want(B)", 0.91)

        TestRule2.inference.check_prob(system.get_state(), "a_u^p", "Ask(B)", 0.91 * 0.9)
        TestRule2.inference.check_prob(system.get_state(), "a_u^p", "Ask(A)", 0.09 * 0.9)
        TestRule2.inference.check_prob(system.get_state(), "a_u^p", "None", 0.1)

        TestRule2.inference.check_prob(system.get_state(), "a_u", "Ask(B)", 0.918)
        TestRule2.inference.check_prob(system.get_state(), "a_u", "None", 0.081)

        EquivalenceDistribution.none_prob = eq_factor
        StatePruner.value_pruning_threshold = old_prune_threshold

    def test_2(self):
        system = DialogueSystem(TestRule2.domain)
        system.get_settings().show_gui = False
        system.detach_module(ForwardPlanner)
        eq_factor = EquivalenceDistribution.none_prob
        EquivalenceDistribution.none_prob = 0.1
        old_prune_threshold = StatePruner.value_pruning_threshold
        StatePruner.value_pruning_threshold = 0.0
        system.start_system()

        TestRule2.inference.check_prob(system.get_state(), "u_u2^p", "Do A", 0.216)
        TestRule2.inference.check_prob(system.get_state(), "u_u2^p", "Please do C", 0.027)
        TestRule2.inference.check_prob(system.get_state(), "u_u2^p", "Could you do B?", 0.054)
        TestRule2.inference.check_prob(system.get_state(), "u_u2^p", "Could you do A?", 0.162)
        TestRule2.inference.check_prob(system.get_state(), "u_u2^p", "none", 0.19)

        builder = CategoricalTableBuilder("u_u2")
        builder.add_row("Please do B", 0.4)
        builder.add_row("Do B", 0.4)

        system.get_state().remove_nodes(system.get_state().get_action_node_ids())
        system.get_state().remove_nodes(system.get_state().get_utility_node_ids())
        system.add_content(builder.build())

        TestRule2.inference.check_prob(system.get_state(), "i_u2", "Want(B)", 0.654)
        TestRule2.inference.check_prob(system.get_state(), "i_u2", "Want(A)", 0.1963)
        TestRule2.inference.check_prob(system.get_state(), "i_u2", "Want(C)", 0.0327)
        TestRule2.inference.check_prob(system.get_state(), "i_u2", "none", 0.1168)

        EquivalenceDistribution.none_prob = eq_factor
        StatePruner.value_pruning_threshold = old_prune_threshold

    def test_3(self):
        system = DialogueSystem(TestRule2.domain)
        system.get_settings().show_gui = False
        system.detach_module(ForwardPlanner)
        eq_factor = EquivalenceDistribution.none_prob
        EquivalenceDistribution.none_prob = 0.1
        old_prune_threshold = StatePruner.value_pruning_threshold
        StatePruner.value_pruning_threshold = 0.0
        system.start_system()

        TestRule2.inference.check_util(system.get_state(), "a_m'", "Do(A)", 0.6)
        TestRule2.inference.check_util(system.get_state(), "a_m'", "Do(B)", -2.6)

        builder = CategoricalTableBuilder("a_u")
        builder.add_row("Ask(B)", 0.8)
        builder.add_row("None", 0.2)
        system.get_state().remove_nodes(system.get_state().get_action_node_ids())
        system.get_state().remove_nodes(system.get_state().get_utility_node_ids())
        system.add_content(builder.build())

        TestRule2.inference.check_util(system.get_state(), "a_m'", "Do(A)", -4.35)
        TestRule2.inference.check_util(system.get_state(), "a_m'", "Do(B)", 2.357)

        EquivalenceDistribution.none_prob = eq_factor
        StatePruner.value_pruning_threshold = old_prune_threshold

    def test_4(self):
        domain2 = XMLDomainReader.extract_domain(TestRule2.domain_file2)
        system2 = DialogueSystem(domain2)
        system2.get_settings().show_gui = False
        system2.detach_module(ForwardPlanner)
        system2.start_system()

        TestRule2.inference.check_util(system2.get_state(), ["a_m3'", "obj(a_m3)'"], Assignment([Assignment("a_m3'", "Do"), Assignment("obj(a_m3)'", "A")]), 0.3)
        TestRule2.inference.check_util(system2.get_state(), ["a_m3'", "obj(a_m3)'"], Assignment([Assignment("a_m3'", "Do"), Assignment("obj(a_m3)'", "B")]), -1.7)
        TestRule2.inference.check_util(system2.get_state(), ["a_m3'", "obj(a_m3)'"], Assignment([Assignment("a_m3'", "SayHi"), Assignment("obj(a_m3)'", "None")]), -0.9)

    def test_5(self):
        domain2 = XMLDomainReader.extract_domain(TestRule2.domain_file3)
        system2 = DialogueSystem(domain2)
        system2.detach_module(ForwardPlanner)
        system2.get_settings().show_gui = False
        system2.start_system()

        TestRule2.inference.check_util(system2.get_state(), ["a_ml'", "a_mg'", "a_md'"], Assignment([Assignment("a_ml'", "SayYes"), Assignment("a_mg'", "Nod"), Assignment("a_md'", "None")]), 2.4)
        TestRule2.inference.check_util(system2.get_state(), ["a_ml'", "a_mg'", "a_md'"], Assignment([Assignment("a_ml'", "SayYes"), Assignment("a_mg'", "Nod"), Assignment("a_md'", "DanceAround")]), -0.6)
        TestRule2.inference.check_util(system2.get_state(), ["a_ml'", "a_mg'", "a_md'"], Assignment([Assignment("a_ml'", "SayYes"), Assignment("a_mg'", "None"), Assignment("a_md'", "None")]), 1.6)

    def test_6(self):
        domain2 = XMLDomainReader.extract_domain(TestRule2.domain_file4)
        system2 = DialogueSystem(domain2)
        system2.detach_module(ForwardPlanner)
        system2.get_settings().show_gui = False
        system2.start_system()
        TestRule2.inference.check_prob(system2.get_state(), "A", ValueFactory.create("[a1,a2]"), 1.0)
        TestRule2.inference.check_prob(system2.get_state(), "a_u", "Request(ball)", 0.5)
