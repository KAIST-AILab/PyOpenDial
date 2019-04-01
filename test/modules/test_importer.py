class TestImporter:
    domain_file = "test/data/domain-woz.xml"
    dialogue_file = "test/data/woz-dialogue.xml"
    domain_file2 = "test/data/example-domain-params.xml"
    dialogue_file2 = "test/data/dialogue.xml"

    # def test_importer(self):
    #     system = DialogueSystem(XMLDomainReader.extract_domain(self.domain_file))
    #
    #     # NEED GUI
    #     # system.get_settings().show_gui = False
    #
    #     Settings.nr_samples = Settings.nr_samples / 10.0
    #     importer = system.import_dialogue(self.dialogue_file)
    #     system.start_system()
    #
    #     while importer.is_alive():
    #         threading.Event().wait(250)
    #
    #     # NEED DialogueRecorder
    #     # self.assertEqual()assertEquals(20, StringUtils.count_occurences(system.get_module(DialogueRecorder).getRecord(), "systemTurn"))
    #     # self.assertEqual(22, StringUtils.count_occurences(system.get_module(DialogueRecorder).getRecord(), "userTurn"))
    #
    #     Settings.nr_samples = Settings.nr_samples * 10

    # def test_importer2(self):
    #     system = DialogueSystem(XMLDomainReader.extract_domain(self.domain_file))
    #     system.get_settings().show_gui = False
    #     Settings.nr_samples = Settings.nr_samples / 5.0
    #     system.start_system()
    #     importer = system.import_dialogue(self.dialogue_file)
    #     importer.setWizardOfOzMode(True)
    #
    #     while importer.is_alive():
    #         threading.Event().wait(300)
    #
    #     # NEED DialogueRecorder
    #     # self.assertEqual(20, StringUtils.count_occurences(system.get_module(DialogueRecorder).getRecord(), "systemTurn"))
    #     # self.assertEqual(22, StringUtils.count_occurences(system.get_module(DialogueRecorder).getRecord(), "userTurn"))
    #
    #     self.assertTrue(system.get_state().getChanceNode("theta_1").get_distrib().get_function().get_mean()[0] > 12.0)
    #     Settings.nr_samples = Settings.nr_samples * 5

    # def test_importer3(self):
    #     system = DialogueSystem(XMLDomainReader.extract_domain(self.domain_file2))
    #     system.get_settings().showGUI = False
    #     system.start_system()
    #     importer = system.import_dialogue(self.dialogue_file2)
    #
    #     while importer.is_alive():
    #         threading.Event().wait(300)
    #
    #     # NEED DialogueRecorder
    #     # self.assertEqual(10, StringUtils.count_occurences(system.get_module(DialogueRecorder).getRecord(), "systemTurn"))
    #     # self.assertEqual(10, StringUtils.count_occurences(system.get_module(DialogueRecorder).getRecord(), "userTurn"))
    #
    #     self.assertAlmostEqual(system.get_state().getChanceNode("theta_repeat").get_distrib().get_function().get_mean()[0], 0.0, delta=0.2)

    # def test_importer4(self):
    #     system = DialogueSystem(XMLDomainReader.extract_domain(self.domain_file2))
    #     Settings.nr_samples = Settings.nr_samples * 3
    #     Settings.max_sampling_time = Settings.max_sampling_time * 3
    # 
    #     # NEED GUI
    #     # system.get_settings().show_gui = False
    # 
    #     system.start_system()
    #     importer = system.import_dialogue(self.dialogue_file2)
    #     importer.setWizardOfOzMode(True)
    # 
    #     while importer.is_alive():
    #         threading.Event().wait(250)
    # 
    #     # NEED DialogueRecorder
    #     # self.assertEqual(10, StringUtils.count_occurences(system.get_module(DialogueRecorder).getRecord(), "systemTurn"))
    #     # self.assertEqual(10, StringUtils.count_occurences(system.get_module(DialogueRecorder).getRecord(), "userTurn"))
    # 
    #     self.assertAlmostEqual(system.get_state().getChanceNode("theta_repeat").get_distrib().get_function().get_mean()[0], 1.35, delta=0.3)
    # 
    #     Settings.nr_samples = Settings.nr_samples / 3.0
    #     Settings.maxSamplingTime = Settings.maxSamplingTime / 3.0
