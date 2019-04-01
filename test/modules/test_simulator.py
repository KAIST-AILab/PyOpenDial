import logging
from time import sleep

import pytest

from dialogue_system import DialogueSystem
from modules.dialogue_recorder import DialogueRecorder
from modules.simulation.simulator import Simulator, XMLDomainReader
from settings import Settings


class TestSimulator:

    # logger
    log = logging.getLogger('PyOpenDial')

    main_domain = "test/data/domain-demo.xml"
    sim_domain = "test/data/domain-simulator.xml"
    main_domain2 = "test/data/example-domain-params.xml"
    sim_domain2 = "test/data/example-simulator.xml"

    # NEED DialogueRecorder
    def test_simulator(self):
        system = DialogueSystem()
        nr_samples = Settings.nr_samples
        Settings.nr_samples = nr_samples / 5.0

        out_break = False
        for k in range(3):
            if out_break:
                break

            system = DialogueSystem(XMLDomainReader.extract_domain(TestSimulator.main_domain))
            if k > 0:
                self.log.warning("restarting the simulator...")
                pass

            system.get_domain().get_models().pop(0)
            system.get_domain().get_models().pop(0)
            system.get_domain().get_models().pop(0)

            sim_domain2 = XMLDomainReader.extract_domain(TestSimulator.sim_domain)
            sim = Simulator(system, sim_domain2)

            system.attach_module(sim)

            # NEED GUI
            system.get_settings().show_gui = False

            system.start_system()

            for i in range(40):
                if system.get_module(Simulator) is None:
                    break
                sleep(0.2)

                # NEED DialogueRecorder
                str = system.get_module(DialogueRecorder).get_record()
                try:
                    self.check_condition(str)
                    system.detach_module(Simulator)
                    out_break = True
                    break
                except:
                    pass

            if not out_break:
                system.detach_module(Simulator)

        # NEED DialogueRecorder
        self.check_condition(system.get_module(DialogueRecorder).get_record())

        system.detach_module(Simulator)
        system.pause(True)
        Settings.nr_samples = nr_samples * 5

    def test_reward_learner(self):
        system = DialogueSystem()
        Settings.nr_samples = Settings.nr_samples * 2

        out_break = False
        for k in range(3):
            if out_break is True:
                break

            if k > 0:
                # log message
                pass

            system = DialogueSystem(XMLDomainReader.extract_domain(TestSimulator.main_domain2))
            sim_domain3 = XMLDomainReader.extract_domain(TestSimulator.sim_domain2)
            sim = Simulator(system, sim_domain3)
            system.attach_module(sim)

            # NEED GUI
            system.get_settings().show_gui = False

            system.start_system()

            for i in range(20):
                if system.get_module(Simulator) is None:
                    break

                sleep(0.1)

                try:
                    self.check_condition2(system)
                    system.detach_module(Simulator)
                    out_break = True
                    break
                except:
                    assert False

            system.detach_module(Simulator)

        self.check_condition2(system)
        system.detach_module(Simulator)
        system.pause(True)
        theta_correct = system.get_content("theta_correct").to_continuous().get_function()
        theta_incorrect = system.get_content("theta_incorrect").to_continuous().get_function()
        theta_repeat = system.get_content("theta_repeat").to_continuous().get_function()
        self.log.debug("theta_correct %s" % theta_correct)
        self.log.debug("theta_incorrect %s" % theta_incorrect)
        self.log.debug("theta_repeat %s" % theta_repeat)

        Settings.nr_samples = Settings.nr_samples / 2.0

    @staticmethod
    def check_condition(str):
        assert "AskRepeat" in str
        assert "Do(Move" in str
        assert "YouSee" in str
        assert "Reward: 10" in str
        assert "Do(Pick" in str

    @staticmethod
    def check_condition2(system):
        theta_correct = system.get_content("theta_correct").to_continuous().get_function()
        theta_incorrect = system.get_content("theta_incorrect").to_continuous().get_function()
        theta_repeat = system.get_content("theta_repeat").to_continuous().get_function()

        assert theta_correct.get_mean()[0] == pytest.approx(2.0, abs=0.7)
        assert theta_incorrect.get_mean()[0] == pytest.approx(-2.0, abs=1.5)
        assert theta_repeat.get_mean()[0] == pytest.approx(0.5, abs=0.7)

