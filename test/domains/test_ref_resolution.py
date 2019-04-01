import pytest

from bn.values.value_factory import ValueFactory
from dialogue_system import DialogueSystem
from modules.state_pruner import StatePruner
from readers.xml_domain_reader import XMLDomainReader


class TestRefResolution:
    domain = XMLDomainReader.extract_domain("test/data/refres.xml")

    def test_nlu(self):
        system = DialogueSystem(TestRefResolution.domain)
        system.get_settings().show_gui = False
        system.start_system()
        system.add_user_input("take the red box")
        assert system.get_content("properties(ref_main)").get_best() == ValueFactory.create("[type=box, def=def, nb=sg, attr=red]")
        system.add_user_input("take the big yellow box")
        assert system.get_content("properties(ref_main)").get_best() == ValueFactory.create("[type=box, def=def, nb=sg, attr=big, attr=yellow]")
        system.add_user_input("take the big and yellow box")
        assert system.get_content("properties(ref_main)").get_best() == ValueFactory.create("[type=box, def=def, nb=sg, attr=big, attr=yellow]")
        system.add_user_input("take the big box on your left")
        assert system.get_content("properties(ref_main)").get_best() == ValueFactory.create("[rel=left(agent), type=box, def=def, nb=sg, attr=big]")
        system.add_user_input("take the big box on the left")
        assert system.get_content("properties(ref_main)").to_discrete().get_prob("[rel=left(agent), type=box, def=def, nb=sg, attr=big]") == pytest.approx(0.5, abs=0.01)
        assert system.get_content("properties(ref_main)").to_discrete().get_prob("[rel=left(spk), type=box, def=def, nb=sg, attr=big]") == pytest.approx(0.5, abs=0.01)
        system.add_user_input("take one box now")
        assert system.get_content("properties(ref_main)").get_best() == ValueFactory.create("[def=indef, nb=sg, type=box]")
        system.add_user_input("take the small and ugly box ")
        assert system.get_content("properties(ref_main)").get_best() == ValueFactory.create("[type=box, def=def, nb=sg, attr=small, attr=ugly]")
        system.add_user_input("now please pick up the book that is behind you")
        assert system.get_content("properties(ref_main)").get_best() == ValueFactory.create("[type=book, def=def, nb=sg, rel=behind(ref_behind)]")
        assert system.get_content("ref_behind").get_best() == ValueFactory.create("you")
        assert system.get_content("ref_main").get_best() == ValueFactory.create("the book")
        system.add_user_input("could you take the red ball on the desk")
        assert system.get_content("properties(ref_main)").get_best() == ValueFactory.create("[type=ball, attr=red, rel=on(ref_on), def=def, nb=sg]")
        assert system.get_content("ref_main").get_best() == ValueFactory.create("the red ball")
        assert system.get_content("ref_on").get_best() == ValueFactory.create("the desk")
        system.add_user_input("could you take the red ball next to the window")
        assert system.get_content("properties(ref_main)").get_best() == ValueFactory.create("[type=ball, attr=red, rel=next to(ref_next to), def=def, nb=sg]")
        assert system.get_content("ref_main").get_best() == ValueFactory.create("the red ball")
        assert system.get_content("ref_next to").get_best() == ValueFactory.create("the window")

        system.add_user_input("could you take the big red ball near the window to your left")
        assert system.get_content("properties(ref_main)").get_best() == ValueFactory.create("[type=ball, attr=red, attr=big, rel=near(ref_near), def=def, nb=sg]")
        assert system.get_content("ref_main").get_best() == ValueFactory.create("the big red ball")
        assert system.get_content("properties(ref_near)").get_best() == ValueFactory.create("[type=window, rel=left(agent), def=def, nb=sg]")
        assert system.get_content("ref_near").get_best() == ValueFactory.create("the window")

        system.add_user_input("could you take the big red ball near the window and to your left")
        assert system.get_content("properties(ref_main)").get_best() == ValueFactory.create("[type=ball, attr=red, attr=big, rel=left(agent), rel=near(ref_near), def=def, nb=sg]")
        assert system.get_content("ref_main").get_best() == ValueFactory.create("the big red ball")
        assert system.get_content("properties(ref_near)").get_best() == ValueFactory.create("[type=window, def=def, nb=sg]")
        assert system.get_content("ref_near").get_best() == ValueFactory.create("the window")

        system.add_user_input("and now pick up the books that are on top of the shelf")
        assert system.get_content("properties(ref_main)").get_best() == ValueFactory.create("[type=book, rel=top(ref_top), def=def, nb=pl]")
        assert system.get_content("properties(ref_top)").get_best() == ValueFactory.create("[type=shelf,def=def, nb=sg]")
        system.add_user_input("and now pick up one book which is big")
        assert system.get_content("properties(ref_main)").get_best() == ValueFactory.create("[type=book,def=indef, attr=big, nb=sg]")

        nbest = dict()
        nbest["and take the red book"] = 0.5
        nbest["and take the wred hook"] = 0.1
        system.add_user_input(nbest)
        assert system.get_content("properties(ref_main)").get_prob("[type=book,attr=red,def=def,nb=sg]") == pytest.approx(0.5, abs=0.01)
        assert system.get_content("properties(ref_main)").get_prob("[type=hook,attr=wred,def=def,nb=sg]") == pytest.approx(0.1, abs=0.01)

    def test_resolution(self):
        system = DialogueSystem(TestRefResolution.domain)
        system.get_settings().show_gui = False
        system.start_system()

        system.add_user_input("take the red ball")
        assert str(system.get_content("a_m").get_best()) == "Select(object_1)"
        assert system.get_content("matches(ref_main)").get_prob("[object_1]") == pytest.approx(0.94, abs=0.05)

        system.add_user_input("take the red object")
        assert str(system.get_content("a_m").get_best()) == "AskConfirm(object_1)"
        assert system.get_content("matches(ref_main)").get_prob("[object_1,object_3]") == pytest.approx(0.39, abs=0.05)
        assert system.get_content("matches(ref_main)").get_prob("[object_1]") == pytest.approx(0.108, abs=0.05)

        system.add_user_input("take the box")
        assert str(system.get_content("a_m").get_best()) == "Select(object_2)"
        assert system.get_content("matches(ref_main)").get_prob("[object_2]") == pytest.approx(0.7, abs=0.05)
        assert system.get_content("matches(ref_main)").get_prob("[]") == pytest.approx(0.3, abs=0.05)

        nbest = dict()
        nbest["and take the ball now"] = 0.3
        system.add_user_input(nbest)
        assert str(system.get_content("a_m").get_best()) == "AskConfirm(object_1)"
        assert system.get_content("matches(ref_main)").get_prob("[object_1]") == pytest.approx(0.27, abs=0.005)
        system.add_user_input("yes")
        assert str(system.get_content("a_m").get_best()) == "Select(object_1)"

        system.add_user_input("pick up the newspaper")
        assert str(system.get_content("a_m").get_best()) == "Failed(the newspaper)"

        system.add_user_input("pick up an object")
        assert str(system.get_content("a_m").get_best()).startswith("AskConfirm(object_")
        assert system.get_content("matches(ref_main)").get_prob("[object_1,object_2,object_3]") == pytest.approx(1.0, abs=0.005)
        system.add_user_input("no")
        assert str(system.get_content("a_m").get_best()).startswith("AskConfirm(object_")
        system.add_user_input("yes")
        assert str(system.get_content("a_m").get_best()).startswith("Select(object_")

        system.add_user_input("pick up the ball to the left of the box")
        assert str(system.get_content("a_m").get_best()) == "Select(object_1)"
        assert system.get_content("matches(ref_main)").get_prob("[object_1]") == pytest.approx(0.75, abs=0.05)

        system.add_user_input("pick up the box to the left of the ball")
        assert str(system.get_content("a_m").get_best()) == "Failed(the box)"
        assert system.get_content("matches(ref_main)").get_prob("[object_2]") == pytest.approx(0.34, abs=0.05)

    def test_underspec(self):
        domain = XMLDomainReader.extract_domain("test/data/underspectest.xml")
        system = DialogueSystem(domain)
        system.get_settings().show_gui = False
        StatePruner.enable_reduction = False
        system.start_system()
        assert system.get_content("match").get_prob("obj_1") == pytest.approx(0.66, abs=0.05)
        assert system.get_content("match").get_prob("obj_3") == pytest.approx(0.307, abs=0.05)
        assert len(system.get_state().get_chance_node_ids()) == 14
        StatePruner.enable_reduction = True
