from copy import copy

import pytest

from datastructs.assignment import Assignment
from dialogue_system import DialogueSystem
from readers.xml_domain_reader import XMLDomainReader


class TestFlightBooking:
    domain = XMLDomainReader.extract_domain("test/data/example-flightbooking.xml")

    def test_dialogue(self):
        system = DialogueSystem(TestFlightBooking.domain)
        system.get_settings().show_gui = False
        system.start_system()
        assert str(system.get_content("u_m").get_best()).find("your destination?") != -1

        u_u = dict()
        u_u["to Bergen"] = 0.4
        u_u["to Bethleem"] = 0.2
        system.add_user_input(u_u)
        assert system.get_content("a_u").get_prob("[Inform(Airport,Bergen)]") == pytest.approx(0.833, abs=0.01)
        assert len(system.get_content("a_u").get_values()) == pytest.approx(3, abs=0.01)
        assert system.get_state().query_prob("a_u", False).get_prob("[Other]") == pytest.approx(0.055, abs=0.01)
        assert str(system.get_content("a_m").to_discrete().get_best()) == "Confirm(Destination,Bergen)"
        u_u.clear()
        u_u["yes exactly"] = 0.8
        system.add_user_input(u_u)
        assert system.get_content("a_u").get_prob("[Confirm]") == pytest.approx(0.98, abs=0.01)
        assert system.get_content("Destination").get_prob("Bergen") == pytest.approx(1.0, abs=0.01)
        assert str(system.get_content("a_m").to_discrete().get_best()) == "Ground(Destination,Bergen)"
        assert str(system.get_content("u_m").get_best()).find("your departure?") != -1
        u_u.clear()
        u_u["to Stockholm"] = 0.8
        system.add_user_input(u_u)
        assert system.get_content("a_u").get_prob("[Other]") == pytest.approx(0.8, abs=0.01)
        assert str(system.get_content("a_m-prev").to_discrete().get_best()) == "Ground(Destination,Bergen)"
        assert len(system.get_content("Destination").to_discrete().get_values()) == 1
        assert system.get_content("Destination").get_prob("Bergen") == pytest.approx(1.0, abs=0.01)
        assert not system.get_state().has_chance_node("Departure")
        assert str(system.get_content("a_m").get_best()) == "AskRepeat"
        assert str(system.get_content("u_m").get_best()).find("you repeat?") != -1
        u_u.clear()
        u_u["to Sandefjord then"] = 0.6
        system.add_user_input(u_u)
        assert system.get_content("a_u").get_prob("None") == pytest.approx(0.149, abs=0.05)
        assert system.get_content("Departure").get_prob("Sandefjord") == pytest.approx(0.88, abs=0.05)
        assert str(system.get_content("a_m").to_discrete().get_best()) == "Confirm(Departure,Sandefjord)"
        assert str(system.get_content("u_m").get_best()).find("that correct?") != -1
        u_u.clear()
        u_u["no to Trondheim sorry"] = 0.08
        system.add_user_input(u_u)
        assert system.get_content("a_u").get_prob("[Inform(Airport,Trondheim),Disconfirm]") == pytest.approx(0.51, abs=0.01)
        assert system.get_content("Departure").get_prob("Trondheim") == pytest.approx(0.51, abs=0.05)
        assert str(system.get_content("a_m").to_discrete().get_best()) == "AskRepeat"
        assert str(system.get_content("u_m").get_best()).find("repeat?") != -1
        u_u.clear()
        u_u["to Trondheim"] = 0.3
        u_u["Sandefjord"] = 0.1
        system.add_user_input(u_u)
        assert system.get_content("a_u").get_prob("[Inform(Airport,Trondheim)]") == pytest.approx(0.667, abs=0.01)
        assert system.get_content("Destination").get_prob("Bergen") == pytest.approx(1.0, abs=0.01)
        assert system.get_content("Departure").get_prob("Trondheim") == pytest.approx(0.89, abs=0.01)
        assert str(system.get_content("a_m").to_discrete().get_best()) == "Confirm(Departure,Trondheim)"
        u_u.clear()
        u_u["yes exactly that's it"] = 0.8
        system.add_user_input(u_u)
        assert str(system.get_content("a_m").to_discrete().get_best()) == "Ground(Departure,Trondheim)"
        assert str(system.get_content("u_m").get_best()).find("which date") != -1
        u_u.clear()
        u_u["that will be on May 26"] = 0.4
        u_u["this will be on May 24"] = 0.2
        system.add_user_input(u_u)
        assert system.get_content("a_u").get_prob("[Inform(Date,May,24)]") == pytest.approx(0.2, abs=0.01)
        assert system.get_content("a_u").get_prob("[Inform(Date,May,26)]") == pytest.approx(0.4, abs=0.01)
        assert system.get_content("Destination").get_prob("Bergen") == pytest.approx(1.0, abs=0.01)
        assert system.get_content("Departure").get_prob("Trondheim") == pytest.approx(1.0, abs=0.01)
        assert system.get_content("Date").get_prob("May 26") == pytest.approx(0.4, abs=0.01)
        assert system.get_content("Date").get_prob("May 24") == pytest.approx(0.2, abs=0.01)
        assert str(system.get_content("a_m").to_discrete().get_best()) == "AskRepeat"
        assert str(system.get_content("a_m-prev").to_discrete().get_best()) == "Ground(Departure,Trondheim)"
        u_u.clear()
        u_u["May 24"] = 0.5
        u_u["Mayday four"] = 0.5
        system.add_user_input(u_u)
        assert system.get_content("a_u").get_prob("[Inform(Date,May,24)]") == pytest.approx(0.82, abs=0.05)
        assert system.get_content("a_u").get_prob("[Inform(Number,4)]") == pytest.approx(0.176, abs=0.01)
        assert system.get_content("Date").get_prob("May 26") == pytest.approx(0.02, abs=0.01)
        assert system.get_content("Date").get_prob("May 24") == pytest.approx(0.94, abs=0.01)
        assert system.get_state().has_chance_node("a_m")
        assert str(system.get_content("a_m-prev").to_discrete().get_best()) == "AskRepeat"
        assert str(system.get_content("a_m").to_discrete().get_best()) == "Ground(Date,May 24)"
        assert str(system.get_content("u_m").get_best()).find("return trip") != -1
        u_u.clear()
        u_u["no thanks"] = 0.9
        system.add_user_input(u_u)
        assert str(system.get_content("u_m").get_best()).find("to order tickets?") != -1
        assert system.get_content("ReturnDate").get_prob("NoReturn") == pytest.approx(1.0, abs=0.01)
        assert system.get_content("current_step").get_prob("MakeOffer") == pytest.approx(1.0, abs=0.01)
        assert str(system.get_content("a_m").get_best()) == "MakeOffer(179)"
        u_u.clear()
        u_u["yes"] = 0.02
        system.add_user_input(u_u)
        assert system.get_content("a_u").get_prob("[Confirm]") == pytest.approx(0.177, abs=0.01)
        assert system.get_content("current_step").get_prob("MakeOffer") == pytest.approx(1.0, abs=0.01)
        assert not system.get_state().has_chance_node("a_m")
        u_u.clear()
        u_u["yes"] = 0.8
        system.add_user_input(u_u)
        assert system.get_content("a_u").get_prob("[Confirm]") == pytest.approx(0.978, abs=0.01)
        assert system.get_content("current_step").get_prob("NbTickets") == pytest.approx(1.0, abs=0.01)
        assert str(system.get_content("u_m").get_best()).find("many tickets") != -1
        u_u.clear()
        u_u["uh I don't know me"] = 0.6
        system.add_user_input(u_u)
        assert system.get_content("a_u").get_prob("[Other]") == pytest.approx(0.6, abs=0.01)
        assert system.get_content("current_step").get_prob("NbTickets") == pytest.approx(1.0, abs=0.01)
        assert str(system.get_content("a_m-prev").to_discrete().get_best()) == "Ground(MakeOffer)"
        assert str(system.get_content("a_m").to_discrete().get_best()) == "AskRepeat"
        u_u.clear()
        u_u["three tickets please"] = 0.9
        system.add_user_input(u_u)
        assert system.get_content("a_u").get_prob("[Inform(Number,3)]") == pytest.approx(0.9, abs=0.01)
        assert system.get_content("current_step").get_prob("NbTickets") == pytest.approx(1.0, abs=0.01)
        assert str(system.get_content("a_m-prev").to_discrete().get_best()) == "AskRepeat"
        assert str(system.get_content("a_m").to_discrete().get_best()) == "Confirm(NbTickets,3)"
        u_u.clear()
        u_u["no sorry two tickets"] = 0.4
        u_u["sorry to tickets"] = 0.3
        system.add_user_input(u_u)
        assert len(system.get_content("a_u").get_values()) == pytest.approx(3, abs=0.01)
        assert system.get_content("a_u").get_prob("[Disconfirm,Inform(Number,2)]") == pytest.approx(0.86, abs=0.05)
        assert system.get_content("NbTickets").get_prob(2) == pytest.approx(0.86, abs=0.05)
        assert system.get_content("NbTickets").get_prob(3) == pytest.approx(0.125, abs=0.05)
        assert str(system.get_content("a_m-prev").to_discrete().get_best()) == "Confirm(NbTickets,3)"
        assert str(system.get_content("a_m").to_discrete().get_best()) == "Confirm(NbTickets,2)"
        assert system.get_content("current_step").get_prob("NbTickets") == pytest.approx(1.0, abs=0.01)
        u_u.clear()
        u_u["yes thank you"] = 0.75
        u_u["yes mind you"] = 0.15
        system.add_user_input(u_u)
        assert len(system.get_content("a_u").get_values()) == pytest.approx(2, abs=0.01)
        assert system.get_content("a_u").get_prob("[Confirm]") == pytest.approx(1.0, abs=0.05)
        assert system.get_content("NbTickets").get_prob(2) == pytest.approx(1.0, abs=0.05)
        assert system.get_content("NbTickets").get_prob(3) == pytest.approx(0.0, abs=0.05)
        assert str(system.get_content("a_m-prev").to_discrete().get_best()) == "Confirm(NbTickets,2)"
        assert str(system.get_content("a_m").to_discrete().get_best()) == "Ground(NbTickets,2)"
        assert system.get_content("current_step").get_prob("LastConfirm") == pytest.approx(1.0, abs=0.01)
        assert str(system.get_content("u_m").get_best()).find("Shall I confirm") != -1
        assert str(system.get_content("u_m").get_best()).find("358 EUR") != -1
        u_u.clear()
        u_u["err yes"] = 0.2
        system.add_user_input(u_u)
        assert system.get_content("a_u").get_prob("[Confirm]") == pytest.approx(0.726, abs=0.01)
        assert system.get_content("current_step").get_prob("LastConfirm") == pytest.approx(1.0, abs=0.01)
        u_u.clear()
        u_u["yes please confirm"] = 0.5
        system.add_user_input(u_u)
        assert system.get_content("a_u").get_prob("[Confirm]") == pytest.approx(0.934, abs=0.01)
        assert system.get_content("current_step").get_prob("Final") == pytest.approx(1.0, abs=0.01)
        assert str(system.get_content("a_m").to_discrete().get_best()) == "Book"
        assert str(system.get_content("u_m").get_best()).find("additional tickets?") != -1
        u_u.clear()
        u_u["thanks but no thanks"] = 0.7
        system.add_user_input(u_u)
        assert system.get_content("a_u").get_prob("[Disconfirm]") == pytest.approx(0.97, abs=0.01)
        assert system.get_content("current_step").get_prob("Close") == pytest.approx(1.0, abs=0.01)
        assert str(system.get_content("u_m").get_best()).find("welcome back!") != -1

        assert sorted(system.get_state().get_chance_node_ids()) == ["Date", "Departure", "Destination", "NbTickets", "ReturnDate", "TotalCost", "a_m", "a_m-prev", "a_u", "current_step", "u_m", "u_u"]
        assert len(system.get_content(["Date", "Departure", "Destination", "NbTickets", "ReturnDate", "TotalCost"]).to_discrete().get_values()) == 1
        assert system.get_content(["Date", "Departure", "Destination", "NbTickets", "ReturnDate", "TotalCost"]).to_discrete().get_prob(
            Assignment([
                Assignment("Date", "May 24"),
                Assignment("Departure", "Trondheim"),
                Assignment("Destination", "Bergen"),
                Assignment("NbTickets", 2.),
                Assignment("ReturnDate", "NoReturn"),
                Assignment("TotalCost", 358.)
            ])
        ) == pytest.approx(1.0, abs=0.01)

    def test_dialogue2(self):
        system = DialogueSystem(TestFlightBooking.domain)
        system.get_settings().show_gui = False
        system.start_system()

        u_u = dict()
        u_u["err I don't know, where can I go?"] = 0.8
        system.add_user_input(u_u)
        assert system.get_content("a_u").to_discrete().get_prob("[Other]") == pytest.approx(0.8, abs=0.01)
        assert str(system.get_content("a_m").get_best()) == "AskRepeat"
        u_u.clear()
        u_u["ah ok well I want to go to Tromsø please"] = 0.8
        system.add_user_input(u_u)
        assert system.get_content("a_u").to_discrete().get_prob("[Inform(Airport,Tromsø)]") == pytest.approx(0.91, abs=0.01)
        assert system.get_content("Destination").to_discrete().get_prob("Tromsø") == pytest.approx(0.91, abs=0.01)
        assert str(system.get_content("a_m").get_best()) == "Confirm(Destination,Tromsø)"
        u_u.clear()
        u_u["that's right"] = 0.6
        system.add_user_input(u_u)
        assert str(system.get_content("a_m").get_best()) == "Ground(Destination,Tromsø)"
        assert str(system.get_content("u_m").get_best()).find("departure?") != -1
        u_u.clear()
        u_u["I'll be leaving from Moss"] = 0.1
        system.add_user_input(u_u)
        assert system.get_content("a_u").to_discrete().get_prob("[Inform(Airport,Moss)]") == pytest.approx(0.357, abs=0.01)
        assert system.get_content("Destination").to_discrete().get_prob("Tromsø") == pytest.approx(1.0, abs=0.01)
        assert system.get_content("Departure").to_discrete().get_prob("Moss") == pytest.approx(0.357, abs=0.01)
        assert str(system.get_content("a_m").get_best()) == "AskRepeat"
        u_u.clear()
        u_u["I am leaving from Moss, did you get that right?"] = 0.2
        u_u["Bodø, did you get that right?"] = 0.4
        system.add_user_input(u_u)
        assert system.get_content("a_u").to_discrete().get_prob("[Confirm,Inform(Airport,Moss)]") == pytest.approx(0.72, abs=0.01)
        assert system.get_content("Departure").to_discrete().get_prob("Moss") == pytest.approx(0.88, abs=0.01)
        assert system.get_content("Departure").to_discrete().get_prob("Bodø") == pytest.approx(0.10, abs=0.01)
        assert str(system.get_content("a_m").get_best()) == "Confirm(Departure,Moss)"
        u_u.clear()
        u_u["yes"] = 0.6
        system.add_user_input(u_u)
        assert str(system.get_content("a_m").get_best()) == "Ground(Departure,Moss)"
        assert str(system.get_content("u_m").get_best()).find("which date") != -1
        u_u.clear()
        u_u["March 16"] = 0.7
        u_u["March 60"] = 0.2
        system.add_user_input(u_u)
        assert system.get_content("a_u").to_discrete().get_prob("[Inform(Date,March,16)]") == pytest.approx(0.7, abs=0.01)
        assert system.get_content("a_u").to_discrete().get_prob("[Other]") == pytest.approx(0.2, abs=0.01)
        assert str(system.get_content("a_m").get_best()) == "AskRepeat"
        u_u.clear()
        u_u["March 16"] = 0.05
        u_u["March 60"] = 0.3
        system.add_user_input(u_u)
        assert str(system.get_content("a_m").get_best()) == "Confirm(Date,March 16)"
        u_u.clear()
        u_u["yes"] = 0.6
        system.add_user_input(u_u)
        assert str(system.get_content("a_m").get_best()) == "Ground(Date,March 16)"
        assert str(system.get_content("u_m").get_best()).find("return trip?") != -1
        u_u.clear()
        u_u["err"] = 0.1
        system.add_user_input(u_u)
        assert not system.get_state().has_chance_node("a_m")
        u_u.clear()
        u_u["yes"] = 0.3
        system.add_user_input(u_u)
        assert str(system.get_content("a_m").get_best()) == "AskRepeat"
        u_u.clear()
        u_u["yes"] = 0.5
        system.add_user_input(u_u)
        assert system.get_content("current_step").get_prob("ReturnDate") == pytest.approx(1.0, abs=0.01)
        assert str(system.get_content("u_m").get_best()).find("travel back") != -1
        u_u.clear()
        u_u["on the 20th of March"] = 0.7
        system.add_user_input(u_u)
        assert str(system.get_content("a_m").get_best()) == "Confirm(ReturnDate,March 20)"
        assert system.get_content("ReturnDate").to_discrete().get_prob("March 20") == pytest.approx(0.7, abs=0.01)
        u_u.clear()
        u_u["yes"] = 0.6
        system.add_user_input(u_u)
        assert str(system.get_content("u_m").get_best()).find("299 EUR") != -1
        assert str(system.get_content("u_m").get_best()).find("to order tickets?") != -1
        copy_state1 = copy(system.get_state())
        u_u.clear()
        u_u["no"] = 0.7
        system.add_user_input(u_u)
        assert str(system.get_content("a_m").get_best()) == "Ground(Cancel)"
        assert str(system.get_content("current_step").get_best()) == "Final"
        assert str(system.get_content("u_m").get_best()).find("additional tickets?") != -1
        copy_state2 = copy(system.get_state())
        assert str(copy_state2.query_prob("current_step").get_best()) == "Final"
        u_u.clear()
        u_u["no"] = 0.7
        system.add_user_input(u_u)
        assert str(copy_state2.query_prob("current_step").get_best()) == "Final"
        assert str(system.get_content("a_m").get_best()) == "Ground(Close)"
        assert str(system.get_content("current_step").get_best()) == "Close"
        assert str(system.get_content("u_m").get_best()).find("welcome back") != -1
        system.get_state().remove_nodes(system.get_state().get_chance_node_ids())
        system.get_state().add_network(copy_state2)
        assert str(copy_state2.query_prob("current_step").get_best()) == "Final"
        u_u.clear()
        u_u["yes"] = 0.7
        system.add_user_input(u_u)
        assert not system.get_state().has_chance_node("Destination")
        assert str(system.get_content("u_m").get_best()).find("destination?") != -1
        assert sorted(system.get_state().get_chance_node_ids()) == ["Destination^p", "a_m-prev", "current_step", "u_m", "u_u"]

        system.add_user_input("Oslo")
        assert str(system.get_content("a_m").get_best()) == "Ground(Destination,Oslo)"
        system.get_state().remove_nodes(system.get_state().get_chance_node_ids())
        system.get_state().add_network(copy_state1)
        u_u.clear()
        u_u["yes"] = 0.8
        system.add_user_input(u_u)

        assert system.get_content("current_step").get_prob("NbTickets") == pytest.approx(1.0, abs=0.01)
        assert str(system.get_content("a_m").to_discrete().get_best()) == "Ground(MakeOffer)"
        u_u.clear()
        u_u["one single ticket"] = 0.9
        system.add_user_input(u_u)
        assert system.get_content("a_u").get_prob("[Inform(Number,1)]") == pytest.approx(0.9, abs=0.01)
        assert system.get_content("current_step").get_prob("NbTickets") == pytest.approx(1.0, abs=0.01)
        assert str(system.get_content("a_m-prev").to_discrete().get_best()) == "Ground(MakeOffer)"
        assert str(system.get_content("a_m").to_discrete().get_best()) == "Confirm(NbTickets,1)"
        u_u.clear()
        u_u["yes thank you"] = 1.0
        system.add_user_input(u_u)
        assert len(system.get_content("a_u").get_values()) == pytest.approx(1, abs=0.01)
        assert system.get_content("a_u").get_prob("[Confirm]") == pytest.approx(1.0, abs=0.05)
        assert system.get_content("NbTickets").get_prob(1.) == pytest.approx(1.0, abs=0.05)
        assert str(system.get_content("a_m-prev").to_discrete().get_best()) == "Confirm(NbTickets,1)"
        assert str(system.get_content("a_m").to_discrete().get_best()) == "Ground(NbTickets,1)"
        assert system.get_content("current_step").get_prob("LastConfirm") == pytest.approx(1.0, abs=0.01)
        assert str(system.get_content("u_m").get_best()).find("Shall I confirm") != -1
        assert str(system.get_content("u_m").get_best()).find("299 EUR") != -1
        u_u.clear()
        u_u["yes please"] = 0.5
        u_u["yellow"] = 0.4
        system.add_user_input(u_u)
        assert system.get_content("a_u").get_prob("[Confirm]") == pytest.approx(0.9397, abs=0.01)
        assert system.get_content("current_step").get_prob("Final") == pytest.approx(1.0, abs=0.01)
        assert str(system.get_content("a_m").to_discrete().get_best()) == "Book"
        assert str(system.get_content("u_m").get_best()).find("additional tickets?") != -1
        copystate3 = copy(system.get_state())
        u_u.clear()
        u_u["thanks but no thanks"] = 0.7
        system.add_user_input(u_u)
        assert system.get_content("a_u").get_prob("[Disconfirm]") == pytest.approx(0.97, abs=0.01)
        assert system.get_content("current_step").get_prob("Close") == pytest.approx(1.0, abs=0.01)
        assert str(system.get_content("u_m").get_best()).find("welcome back!") != -1
        assert sorted(system.get_state().get_chance_node_ids()) == ["Date", "Departure", "Destination", "NbTickets", "ReturnDate", "TotalCost", "a_m", "a_m-prev", "a_u", "current_step", "u_m", "u_u"]
        assert len(system.get_content(["Date", "Departure", "Destination", "NbTickets", "ReturnDate", "TotalCost"]).to_discrete().get_values()) == 1
        assert system.get_content(["Date", "Departure", "Destination", "NbTickets", "ReturnDate", "TotalCost"]).to_discrete().get_prob(
            Assignment([
                Assignment("Date", "March 16"),
                Assignment("Departure", "Moss"),
                Assignment("Destination", "Tromsø"),
                Assignment("NbTickets", 1.),
                Assignment("ReturnDate", "March 20"),
                Assignment("TotalCost", 299.)
            ])
        ) == pytest.approx(1.0, abs=0.01)

        system.get_state().remove_nodes(system.get_state().get_chance_node_ids())  ##
        system.get_state().add_network(copystate3)
        u_u.clear()
        u_u["yes"] = 0.7
        system.add_user_input(u_u)
        assert not system.get_state().has_chance_node("Destination")
        assert str(system.get_content("u_m").get_best()).find("destination?") != -1
        assert sorted(system.get_state().get_chance_node_ids()) == ["Destination^p", "a_m-prev", "current_step", "u_m", "u_u"]
        system.add_user_input("Oslo")
        assert str(system.get_content("a_m").get_best()) == "Ground(Destination,Oslo)"
