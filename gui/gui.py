# Imports for PyQt5 Lib and Functions to be used
import sys
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import QWidget, QApplication, QPushButton, qApp, QAction, QFileDialog, QMenu, QActionGroup, QDesktopWidget

# alignment to PyQt Widgets
# from PySide2.QtWidgets import QMainWindow, QAction

from bn.distribs.distribution_builder import CategoricalTableBuilder
from datastructs.assignment import Assignment
from modules.audio_module import AudioModule
from modules.forward_planner import ForwardPlanner
from modules.mcts_planner import MCTSPlanner
from readers.xml_domain_reader import XMLDomainReader
from utils.string_utils import StringUtils
from utils.xml_utils import XMLUtils

setStyleQte = """QTextEdit {
    font-family: "Courier"; 
    font-size: 12pt; 
    font-weight: 600; 
    text-align: right;
    background-color: Gainsboro;
}"""

setStyletui = """QLineEdit {
    font-family: "Courier";
    font-weight: 600; 
    text-align: left;
    background-color: Gainsboro;
}"""


class GUI(QtWidgets.QWidget):
    def __init__(self, system):
        super(QtWidgets.QWidget, self).__init__()
        self._system = system
        self._audio_module = None

        self.pr_fun = self.pause_interaction

        self.v = None
        self.layout = QtWidgets.QVBoxLayout(self)
        self.speech = QtWidgets.QPushButton('Press and hold to record speech')
        self.speech.pressed.connect(self.start_recording)
        self.speech.released.connect(self.stop_recording)

        self.font = QFont()
        self.font.setPointSize(13)
        self.chatlog = QtWidgets.QTextEdit()
        self.userinput = QtWidgets.QLineEdit()
        self.userinput.returnPressed.connect(self.AddToChatLogUser)

        self.sendBtn = QtWidgets.QPushButton('Send')
        self.sendBtn.clicked.connect(self.AddToChatLogUser)

        # Menubar
        self.create_menu()

        # GUI Setup
        self.GuiSetup()

        self.setWindowTitle('PyOpenDial toolkit')
        self.setGeometry(1000, 100, 800, 500)
        self.show()

        # if not self._system._domain._xml_file is None:
        #     self.open_domain(self._system._domain)

    def create_menu(self):
        self.menu = QtWidgets.QMainWindow()
        self.statusbar = self.menu.statusBar()
        self.menubar = self.menu.menuBar()
        self.menubar.setNativeMenuBar(False)

        self.add_domain_tab()
        self.add_interaction_tab()
        self.add_options_tab()
        self.add_help_tab()

    def add_domain_tab(self):
        # New
        new = QAction('New', self)
        new.setStatusTip('New')

        # Open file
        open_file = QAction('Open File', self)
        open_file.setStatusTip('Open File')
        open_file.triggered.connect(self.open_domain)

        # Save
        save = QAction('Save', self)
        save.setShortcut('Ctrl+S')
        save.setStatusTip('Save')
        save.setEnabled(False)

        # Save as
        self.save_as = QAction('Save As...', self)
        self.save_as.setStatusTip('Save As...')
        self.save_as.setEnabled(False)

        # Import
        self.import_menu = QMenu('Import', self)
        self.import_act1 = QAction('Dialogue State', self)
        self.import_act2 = QAction('Parameters', self)
        self.import_menu.addAction(self.import_act1)
        self.import_menu.addAction(self.import_act2)
        self.import_act1.triggered.connect(self.import_dialog)

        # Export
        self.export_menu = QMenu('Export', self)
        self.export_act1 = QAction('Dialogue State', self)
        self.export_act2 = QAction('Parameters', self)
        self.export_menu.addAction(self.export_act1)
        self.export_menu.addAction(self.export_act2)
        self.export_act1.setEnabled(False)
        self.export_act2.setEnabled(False)
        self.export_act1.triggered.connect(self.export_dialog)

        # Close
        close = QAction('Close PyOpenDial', self)
        close.setStatusTip('Close PyOpenDial')
        close.triggered.connect(qApp.quit)

        # Domain
        domain = self.menubar.addMenu('Domain')
        domain.addAction(new)
        domain.addAction(open_file)
        domain.addAction(save)
        domain.addAction(self.save_as)
        domain.addMenu(self.import_menu)
        domain.addMenu(self.export_menu)
        domain.addAction(close)

    def add_interaction_tab(self):
        # Reset
        self.reset = QAction('Reset', self)
        self.reset.setShortcut('Ctrl+R')
        self.reset.setStatusTip('Reset')
        self.reset.setEnabled(False)
        self.reset.triggered.connect(self.reset_interaction)

        # Pause/Resume
        self.pause = QAction('Pause/Resume', self)
        self.pause.setShortcut('Ctrl+P')
        self.pause.setStatusTip('Pause/Resume')
        self.pause.setEnabled(False)
        # self.pause.triggered.connect(self.pause_interaction if not self._system._paused else self.resume_interaction)
        self.pause.triggered.connect(self.pr_fun)

        # if not self._system._paused:
        #     self.pause.triggered.connect(self.pause_interaction if not self._system._paused else self.resume_interaction)
        # else:
        #     self.pause.triggered.connect(self.resume_interaction)

        # Connect to Remote Client
        crc = QAction('Connect to Remote Client', self)
        crc.setStatusTip('Connect to Remote Client')

        # interaction role
        interact_menu = QMenu('Interaction role', self)

        group = QActionGroup(interact_menu)

        user = QAction('User', self, checkable=True)
        system = QAction('System', self, checkable=True)
        interact_menu.addAction(user)
        interact_menu.addAction(system)
        group.addAction(user)
        group.addAction(system)
        group.setExclusive(True)

        # Import Dialogue From...
        import_dialog = QMenu('Import Dialogue From...', self)

        normal = QAction('Normal Transcript', self)
        woz = QAction('Wizard-of-Oz Transcript', self)
        import_dialog.addAction(normal)
        import_dialog.addAction(woz)

        # Save Dialogue As...
        save = QAction('Save Dialogue As...', self)

        # Interaction
        interaction = self.menubar.addMenu('Interaction')
        interaction.addAction(self.reset)
        interaction.addAction(self.pause)
        interaction.addAction(crc)
        interaction.addMenu(interact_menu)
        interaction.addMenu(import_dialog)
        interaction.addAction(save)

    def add_options_tab(self):
        # Planner
        planner_menu = QMenu('Planner', self)
        forward_planner = QAction('Forward Planner', self)
        mcts_planner = QAction('MCTS Planner', self)
        planner_menu.addAction(forward_planner)
        planner_menu.addAction(mcts_planner)
        forward_planner.triggered.connect(self.switch_forward)
        mcts_planner.triggered.connect(self.switch_mcts)

        options = self.menubar.addMenu('Options')
        options.addMenu(planner_menu)

    def add_help_tab(self):
        # New
        about = QAction('About', self)
        about.setStatusTip('About')

        # New
        document = QAction('Documentation', self)
        document.setStatusTip('Documentation')

        help = self.menubar.addMenu('Help')
        help.addAction(about)
        help.addAction(document)

    def open_domain(self, domain=False):
        if domain is False:
            fname = QFileDialog.getOpenFileName(self, 'Open File')
            domain = XMLDomainReader.extract_domain(fname[0])

            domain_name = str(domain).split('.')[0]

            self.chatlog.clear()
            self.chatlog.append("[%s domain successfully attached]\n" % domain_name)
            self.chatlog.setAlignment(Qt.AlignLeft)
            self._system.change_domain(domain)

            self.save_as.setEnabled(True)
            self.reset.setEnabled(True)
            self.pause.setEnabled(True)
            self.export_act1.setEnabled(True)

        else:
            self._system.change_domain(domain)

    def reset_interaction(self):
        self.chatlog.clear()
        self.chatlog.append("[Reinitializing interaction...]\n")
        self.chatlog.setAlignment(Qt.AlignLeft)
        self._system.change_domain(self._system.get_domain())

    def pr_interaction(self):
        if self._system._paused:
            return self.resume_interaction
        else:
            return self.pause_interaction

    def pause_interaction(self):
        self.chatlog.append("[System paused]\n")
        self.chatlog.setAlignment(Qt.AlignLeft)
        self.chatlog.moveCursor(QtGui.QTextCursor.End)
        self._system._paused = True
        self.pr_fun = self.resume_interaction

    def resume_interaction(self):
        self.chatlog.append("[System resumed]\n")
        self.chatlog.setAlignment(Qt.AlignLeft)
        self.chatlog.moveCursor(QtGui.QTextCursor.End)
        self._system._paused = False
        self.pr_fun = self.pause_interaction

    def export_dialog(self):
        fname = QFileDialog.getSaveFileName(self, 'Save File')
        XMLUtils.export_content(self._system, fname[0], 'state')

    def import_dialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open File')
        importer = self._system.import_dialogues(fname[0])

        self.chatlog.clear()
        self.chatlog.append("[Module FlightBookingExample successfully attached]\n")
        self.chatlog.setAlignment(Qt.AlignLeft)
        self.chatlog.moveCursor(QtGui.QTextCursor.End)
        # self._system.change_domain(domain)

        self.save_as.setEnabled(True)
        self.reset.setEnabled(True)
        self.pause.setEnabled(True)

    def import_params(self):
        pass

    def has_forward_planner(self):
        for module in self._system._modules:
            if isinstance(module, ForwardPlanner):
                return True
        return False

    def has_mcts_planner(self):
        for module in self._system._modules:
            if isinstance(module, MCTSPlanner):
                return True
        return False

    def switch_forward(self):
        self.chatlog.append("[Switch to Forward Planner]\n")
        self.chatlog.setAlignment(Qt.AlignLeft)
        self.chatlog.moveCursor(QtGui.QTextCursor.End)

        if self.has_mcts_planner():
            self._system.detach_module(MCTSPlanner)

        if not self.has_forward_planner():
            self._system.attach_module(ForwardPlanner)

    def switch_mcts(self):
        self.chatlog.append("[Switch to MCTS Planner]\n")
        self.chatlog.setAlignment(Qt.AlignLeft)
        self.chatlog.moveCursor(QtGui.QTextCursor.End)

        if self.has_forward_planner():
            self._system.detach_module(ForwardPlanner)

        if not self.has_mcts_planner():
            self._system.attach_module(MCTSPlanner)

    def record_speech(self):
        pass

    def GuiSetup(self):
        self.layout.addWidget(self.menubar)
        self.chatlog.setStyleSheet(setStyletui)
        self.chatlog.setFont(self.font)
        self.userinput.setStyleSheet(setStyletui)
        self.userinput.setFont(self.font)
        self.speech.setFont(self.font)
        self.layout.addWidget(self.speech)
        self.layout.addWidget(self.chatlog)
        self.layout.addWidget(self.userinput)
        self.layout.addWidget(self.sendBtn)
        self.speech.clicked.connect(self.record_speech)

    def AddToChatLogUser(self):
        umsg = self.userinput.text().strip()
        if self._system._paused:
            self.userinput.setText("")
        else:
            self.userinput.setText("")

            if umsg is '':
                return

            if '/' in umsg:
                self._add_incremental_utterance(umsg)
            elif '=' in umsg:
                self._add_special_input(umsg)
            else:
                self._system.add_user_input(StringUtils.get_table_from_input(umsg))

    def _add_incremental_utterance(self, text):
        follow_previous = text.startswith('/')
        incomplete = text.endswith('/')
        text = text.replace('/', '').strip()

        table = StringUtils.get_table_from_input(text)

        # TODO: Use Thread
        self._system.add_content(Assignment(self._system.get_settings().user_speech, 'busy' if incomplete else 'None'))
        self._system.add_incremental_user_input(table, follow_previous)
        if not incomplete:
            self._system.get_state().set_as_committed(self._system.get_settings.user_input)

    def _add_special_input(self, text):
        special_input = text.split('=')[0].strip()
        text = text.split('=')[1].strip()

        table = StringUtils.get_table_from_input(text)

        builder = CategoricalTableBuilder(special_input)
        for key, value in table.items():
            builder.add_row(key, value)

        self._system.add_content(builder.build())

    def start_recording(self):
        if self._audio_module is None:
            return

        self._audio_module.start_recording()

    def stop_recording(self):
        if self._audio_module is None:
            return

        self._audio_module.stop_recording()

    def enable_speech(self, to_enable):
        if to_enable:
            self._audio_module = self._system.get_module(AudioModule)
        else:
            self._audio_module = None
