from copy import copy

from dialogue_system import DialogueSystem
from domains.rules.effects.effect import Effect
from modules.forward_planner import ForwardPlanner
from modules.state_pruner import StatePruner
from readers.xml_domain_reader import XMLDomainReader
from test.common.inference_checks import InferenceChecks


class TestDialogueState:
    domain_file = "test/data/domain1.xml"

    domain = XMLDomainReader.extract_domain(domain_file)
    inference = InferenceChecks()

    def test_state_copy(self):
        system = DialogueSystem(TestDialogueState.domain)
        system.detach_module(ForwardPlanner)
        StatePruner.enable_reduction = False

        system.get_settings().show_gui = False
        system.start_system()

        initial_state = copy(system.get_state())

        rule_id = ""
        for id in system.get_state().get_node("u_u2").get_output_node_ids():
            if str(system.get_content(id)).find("+=HowAreYou") != -1:
                rule_id = id

        TestDialogueState.inference.check_prob(initial_state, rule_id, Effect.parse_effect("a_u2+=HowAreYou"), 0.9)
        TestDialogueState.inference.check_prob(initial_state, rule_id, Effect.parse_effect("Void"), 0.1)

        TestDialogueState.inference.check_prob(initial_state, "a_u2", "[HowAreYou]", 0.2)
        TestDialogueState.inference.check_prob(initial_state, "a_u2", "[Greet, HowAreYou]", 0.7)
        TestDialogueState.inference.check_prob(initial_state, "a_u2", "[]", 0.1)

        StatePruner.enable_reduction = True

    def test_state_copy2(self):
        InferenceChecks.exact_threshold = 0.08

        system = DialogueSystem(TestDialogueState.domain)
        system.get_settings().show_gui = False
        system.detach_module(ForwardPlanner)
        system.start_system()

        initial_state = copy(system.get_state())

        TestDialogueState.inference.check_prob(initial_state, "a_u2", "[HowAreYou]", 0.2)
        TestDialogueState.inference.check_prob(initial_state, "a_u2", "[Greet, HowAreYou]", 0.7)
        TestDialogueState.inference.check_prob(initial_state, "a_u2", "[]", 0.1)
