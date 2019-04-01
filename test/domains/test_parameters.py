import pytest

from bn.values.value_factory import ValueFactory
from datastructs.assignment import Assignment
from dialogue_system import DialogueSystem
from domains.rules.effects.basic_effect import BasicEffect
from domains.rules.effects.effect import Effect
from domains.rules.parameters.complex_parameter import ComplexParameter
from domains.rules.parameters.single_parameter import SingleParameter
from inference.approximate.sampling_algorithm import SamplingAlgorithm
from modules.forward_planner import ForwardPlanner
from readers.xml_domain_reader import XMLDomainReader
from readers.xml_state_reader import XMLStateReader
from test.common.inference_checks import InferenceChecks


class TestParameters:
    domain_file = "test/data/testwithparams.xml"
    domain_file2 = "test/data/testwithparams2.xml"
    param_file = "test/data/params.xml"

    params = XMLStateReader.extract_bayesian_network(param_file, "parameters")

    domain1 = XMLDomainReader.extract_domain(domain_file)
    domain1.set_parameters(params)

    domain2 = XMLDomainReader.extract_domain(domain_file2)
    domain2.set_parameters(params)

    inference = InferenceChecks()

    def test_param_1(self):
        InferenceChecks.exact_threshold = 0.1

        system = DialogueSystem(TestParameters.domain1)
        system.detach_module(ForwardPlanner)
        system.get_settings().show_gui = False
        assert system.get_state().has_chance_node("theta_1")
        TestParameters.inference.check_cdf(system.get_state(), "theta_1", 0.5, 0.5)
        TestParameters.inference.check_cdf(system.get_state(), "theta_1", 5., 0.99)

        TestParameters.inference.check_cdf(system.get_state(), "theta_2", 1., 0.07)
        TestParameters.inference.check_cdf(system.get_state(), "theta_2", 2., 0.5)

        system.start_system()
        system.add_content("u_u", "hello there")
        utils = ((SamplingAlgorithm()).query_util(system.get_state(), "u_m'"))
        assert utils.get_util(Assignment("u_m'", "yeah yeah talk to my hand")) > 0
        assert utils.get_util(Assignment("u_m'", "so interesting!")) > 1.7
        assert utils.get_util(Assignment("u_m'", "yeah yeah talk to my hand")) < utils.get_util(Assignment("u_m'", "so interesting!"))
        assert len(system.get_state().get_node_ids()) == 11

    def test_param_2(self):
        InferenceChecks.exact_threshold = 0.1

        system = DialogueSystem(TestParameters.domain1)
        system.detach_module(ForwardPlanner)
        system.get_settings().show_gui = False

        assert system.get_state().has_chance_node("theta_3")
        TestParameters.inference.check_cdf(system.get_state(), "theta_3", 0.6, 0.0)
        TestParameters.inference.check_cdf(system.get_state(), "theta_3", 0.8, 0.5)
        TestParameters.inference.check_cdf(system.get_state(), "theta_3", 0.95, 1.0)

        system.start_system()
        system.add_content("u_u", "brilliant")
        distrib = system.get_content("a_u")

        assert distrib.get_prob("approval") == pytest.approx(0.8, abs=0.05)

    def test_param_3(self):
        system = DialogueSystem(TestParameters.domain1)
        system.detach_module(ForwardPlanner)
        system.get_settings().show_gui = False
        system.start_system()

        rules = TestParameters.domain1.get_models()[0].get_rules()
        outputs = rules[1].get_output(Assignment("u_u", "no no"))
        o = Effect(BasicEffect("a_u", "disapproval"))
        assert isinstance(outputs.get_parameter(o), SingleParameter)
        input = Assignment("theta_4", ValueFactory.create("[0.36, 0.64]"))
        assert outputs.get_parameter(o).get_value(input) == pytest.approx(0.64, abs=0.01)

        system.get_state().remove_nodes(system.get_state().get_action_node_ids())
        system.get_state().remove_nodes(system.get_state().get_utility_node_ids())
        system.add_content("u_u", "no no")
        assert system.get_state().query_prob("theta_4").to_continuous().get_function().get_mean()[0] == pytest.approx(0.36, abs=0.1)

        assert system.get_content("a_u").get_prob("disapproval") == pytest.approx(0.64, abs=0.1)

    def test_param_4(self):
        system = DialogueSystem(TestParameters.domain1)
        system.detach_module(ForwardPlanner)
        system.get_settings().show_gui = False
        system.start_system()

        rules = TestParameters.domain1.get_models()[1].get_rules()
        outputs = rules[0].get_output(Assignment("u_u", "my name is"))
        o = Effect(BasicEffect("u_u^p", "Pierre"))
        assert isinstance(outputs.get_parameter(o), SingleParameter)
        input = Assignment("theta_5", ValueFactory.create("[0.36, 0.24, 0.40]"))
        assert outputs.get_parameter(o).get_value(input) == pytest.approx(0.36, abs=0.01)

        system.get_state().remove_nodes(system.get_state().get_action_node_ids())
        system.get_state().remove_nodes(system.get_state().get_utility_node_ids())
        system.add_content("u_u", "my name is")

        system.get_state().remove_nodes(system.get_state().get_action_node_ids())
        system.get_state().remove_nodes(system.get_state().get_utility_node_ids())
        system.add_content("u_u", "Pierre")

        system.get_state().remove_nodes(system.get_state().get_action_node_ids())
        system.get_state().remove_nodes(system.get_state().get_utility_node_ids())
        system.add_content("u_u", "my name is")

        system.get_state().remove_nodes(system.get_state().get_action_node_ids())
        system.get_state().remove_nodes(system.get_state().get_utility_node_ids())
        system.add_content("u_u", "Pierre")

        assert system.get_state().query_prob("theta_5").to_continuous().get_function().get_mean()[0] == pytest.approx(0.3, abs=0.12)

    def test_param_5(self):
        system = DialogueSystem(TestParameters.domain2)
        system.detach_module(ForwardPlanner)
        system.get_settings().show_gui = False
        system.start_system()

        rules = TestParameters.domain2.get_models()[0].get_rules()
        outputs = rules[0].get_output(Assignment("u_u", "brilliant"))
        o = Effect(BasicEffect("a_u", "approval"))
        assert isinstance(outputs.get_parameter(o), ComplexParameter)
        input = Assignment([Assignment("theta_6", 2.1), Assignment("theta_7", 1.3)])
        assert outputs.get_parameter(o).get_value(input) == pytest.approx(0.74, abs=0.01)

        system.get_state().remove_nodes(system.get_state().get_action_node_ids())
        system.get_state().remove_nodes(system.get_state().get_utility_node_ids())
        system.add_content("u_u", "brilliant")

        assert system.get_state().query_prob("theta_6").to_continuous().get_function().get_mean()[0] == pytest.approx(1.0, abs=0.08)

        assert system.get_content("a_u").get_prob("approval") == pytest.approx(0.63, abs=0.08)
        assert system.get_content("a_u").get_prob("irony") == pytest.approx(0.3, abs=0.08)

    def test_param_6(self):
        system = DialogueSystem(XMLDomainReader.extract_domain("test/data/testparams3.xml"))
        system.get_settings().show_gui = False
        system.start_system()
        table = system.get_content("b").to_discrete()
        assert len(table) == 6
        assert table.get_prob("something else") == pytest.approx(0.45, abs=0.05)
        assert table.get_prob("value: first with type 1") == pytest.approx(0.175, abs=0.05)
        assert table.get_prob("value: second with type 2") == pytest.approx(0.05, abs=0.05)
