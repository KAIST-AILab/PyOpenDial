from bn.distribs.distribution_builder import CategoricalTableBuilder
from dialogue_system import DialogueSystem
from modules.forward_planner import ForwardPlanner
from readers.xml_domain_reader import XMLDomainReader
from readers.xml_state_reader import XMLStateReader
from settings import Settings


class TestLearning:
    domain_file = "test/data/domain-is.xml"
    parameters_file = "test/data/params-is.xml"

    def test_IS2013(self):
        domain = XMLDomainReader.extract_domain(TestLearning.domain_file)
        params = XMLStateReader.extract_bayesian_network(TestLearning.parameters_file, "parameters")
        domain.set_parameters(params)
        system = DialogueSystem(domain)
        system.get_settings().show_gui = False
        system.detach_module(ForwardPlanner)
        Settings.nr_samples = Settings.nr_samples * 3
        Settings.max_sampling_time = Settings.max_sampling_time * 10
        system.start_system()

        init_mean = system.get_content("theta_1").to_continuous().get_function().get_mean()

        builder = CategoricalTableBuilder("a_u")
        builder.add_row("Move(Left)", 1.0)
        builder.add_row("Move(Right)", 0.0)
        builder.add_row("None", 0.0)
        system.add_content(builder.build())
        system.get_state().remove_nodes(system.get_state().get_utility_node_ids())
        system.get_state().remove_nodes(system.get_state().get_action_node_ids())

        after_mean = system.get_content("theta_1").to_continuous().get_function().get_mean()

        assert after_mean[0] - init_mean[0] > 0.04
        assert after_mean[1] - init_mean[1] < 0.04
        assert after_mean[2] - init_mean[2] < 0.04
        assert after_mean[3] - init_mean[3] < 0.04
        assert after_mean[4] - init_mean[4] < 0.04
        assert after_mean[5] - init_mean[5] < 0.04
        assert after_mean[6] - init_mean[6] < 0.04
        assert after_mean[7] - init_mean[7] < 0.04

        Settings.nr_samples = int(Settings.nr_samples / 3)
        Settings.max_sampling_time = Settings.max_sampling_time / 10
