from bn.distribs.distribution_builder import CategoricalTableBuilder
from bn.distribs.single_value_distribution import SingleValueDistribution
from dialogue_system import DialogueSystem
from modules.forward_planner import ForwardPlanner
from readers.xml_domain_reader import XMLDomainReader
from readers.xml_state_reader import XMLStateReader


class TestDemo:
    domain_file = "test/data/thesistest.xml"
    param_file = "test/data/thesisparams.xml"
    domain_file2 = "test/data/domain-demo.xml"

    def test_param1(self):
        domain = XMLDomainReader.extract_domain(TestDemo.domain_file)
        params = XMLStateReader.extract_bayesian_network(TestDemo.param_file, "parameters")
        domain.set_parameters(params)
        system = DialogueSystem(domain)
        system.get_settings().show_gui = False

        system.detach_module(ForwardPlanner)
        system.get_settings().show_gui = False

        system.start_system()
        system.add_content("a_m", "AskRepeat")

        t = CategoricalTableBuilder("a_u")
        t.add_row("DoA", 0.7)
        t.add_row("a_u", 0.2)
        t.add_row("a_u", 0.1)
        system.add_content(t.build())
        for i in range(3000):
            print((system.get_state().get_chance_node("theta").sample()).get_array()[0])

    def test_demo(self):
        domain = XMLDomainReader.extract_domain(TestDemo.domain_file2)
        system = DialogueSystem(domain)
        system.get_settings().show_gui = False

        system.start_system()
        assert len(system.get_state().get_chance_nodes()) == 5

        t = CategoricalTableBuilder("u_u")
        t.add_row("hello there", 0.7)
        t.add_row("hello", 0.2)
        updates = system.add_content(t.build())
        assert updates.issuperset({"a_u", "a_m", "u_m"})

        assert str(system.get_content("u_m").get_best()) == "Hi there"

        t2 = dict()
        t2["move forward"] = 0.06
        system.add_user_input(t2)

        assert not system.get_state().has_chance_node("u_m")

        t2 = dict()
        t2["move forward"] = 0.45
        system.add_user_input(t2)

        assert str(system.get_content("u_m").get_best()) == "OK, moving Forward"

        t = CategoricalTableBuilder("u_u")
        t.add_row("now do that again", 0.3)
        t.add_row("move backward", 0.22)
        t.add_row("move a bit to the left", 0.22)
        system.add_content(t.build())

        assert str(system.get_content("u_m").get_best()) == "Sorry, could you repeat?"

        t = CategoricalTableBuilder("u_u")
        t.add_row("do that one more time", 0.65)
        system.add_content(t.build())

        assert str(system.get_content("u_m").get_best()) == "OK, moving Forward"

        system.add_content(SingleValueDistribution("perceived", "[BlueObj]"))

        t = CategoricalTableBuilder("u_u")
        t.add_row("what do you see", 0.6)
        t.add_row("do you see it", 0.3)
        system.add_content(t.build())
        assert str(system.get_content("u_m").get_best()) == "I see a blue cylinder"

        t = CategoricalTableBuilder("u_u")
        t.add_row("pick up the blue object", 0.75)
        t.add_row("turn left", 0.12)
        system.add_content(t.build())

        assert str(system.get_content("u_m").get_best()) == "OK, picking up the blue object"

        system.add_content(SingleValueDistribution("perceived", "[]"))
        system.add_content(SingleValueDistribution("carried", "[BlueObj]"))

        t = CategoricalTableBuilder("u_u")
        t.add_row("now please move a bit forward", 0.21)
        t.add_row("move backward a little bit", 0.13)
        system.add_content(t.build())

        assert str(system.get_content("u_m").get_best()) == "Should I move a bit forward?"

        t = CategoricalTableBuilder("u_u")
        t.add_row("yes", 0.8)
        t.add_row("move backward", 0.1)
        system.add_content(t.build())

        assert str(system.get_content("u_m").get_best()) == "OK, moving Forward a little bit"

        t = CategoricalTableBuilder("u_u")
        t.add_row("and now move forward", 0.21)
        t.add_row("move backward", 0.09)
        system.add_content(t.build())

        assert str(system.get_content("u_m").get_best()) == "Should I move forward?"

        t = CategoricalTableBuilder("u_u")
        t.add_row("no", 0.6)
        system.add_content(t.build())

        assert str(system.get_content("u_m").get_best()) == "Should I move backward?"

        t = CategoricalTableBuilder("u_u")
        t.add_row("yes", 0.5)
        system.add_content(t.build())

        assert str(system.get_content("u_m").get_best()) == "OK, moving Backward"

        t = CategoricalTableBuilder("u_u")
        t.add_row("now what can you see now?", 0.7)
        system.add_content(t.build())

        assert str(system.get_content("u_m").get_best()) == "I do not see anything"

        t = CategoricalTableBuilder("u_u")
        t.add_row("please release the object", 0.5)
        system.add_content(t.build())

        assert str(system.get_content("u_m").get_best()) == "OK, putting down the object"

        t = CategoricalTableBuilder("u_u")
        t.add_row("something unexpected", 0.7)
        system.add_content(t.build())

        assert not system.get_state().has_chance_node("u_m")

        t = CategoricalTableBuilder("u_u")
        t.add_row("goodbye", 0.7)
        system.add_content(t.build())

        assert str(system.get_content("u_m").get_best()) == "Bye, see you next time"
