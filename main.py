"""
Starts the dialogue system. Command-line parameters can specified through system properties via
the --flag. All parameters are optional.

Some of the possible properties are:
--domain path/to/domain/file: dialogue domain file
--dialogue path/to/recorded/dialogue: dialogue file to import
--simulator path/to/simulator/file: domain file for the simulator
"""
import argparse
import logging
import os

from dialogue_system import DialogueSystem
from modules.simulation.simulator import Simulator
from plugins.GoogleSTT import GoogleSTT
from plugins.GoogleTTS import GoogleTTS
from readers.xml_domain_reader import XMLDomainReader

parser = argparse.ArgumentParser()
parser.add_argument('--domain', type=str, help='domain file path')
# parser.add_argument('--dialogue', type=str, help='dialogue file path')
parser.add_argument('--simulator', type=str, help='simulator file path')
args = parser.parse_args()

# Set logger
logger = logging.getLogger('PyOpenDial')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

if args.domain is not None:
    system = DialogueSystem(args.domain)
else:
    system = DialogueSystem()

if args.domain:
    try:
        domain = XMLDomainReader.extract_domain(args.domain)
        system.log.info("Domain from %s successfully extracted" % args.domain)
    except Exception as e:
        system.display_comment("Cannot load domain: %s" % e)
        domain = XMLDomainReader.extract_empty_domain(args.domain)
    system.change_domain(domain)

if args.simulator:
    simulator = Simulator(system, XMLDomainReader.extract_domain(args.simulator))
    system.log.info("Simulator with domain %s successfully extracted" % args.simulator)
    system.attach_module(simulator)

settings = system.get_settings()
system.change_settings(settings)

if os.path.exists(settings.GOOGLE_APPLICATION_CREDENTIALS):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.GOOGLE_APPLICATION_CREDENTIALS
    system.attach_module(GoogleSTT(system))
    system.attach_module(GoogleTTS(system))
    print('Google SST/TTS modules are attached.')
else:
    print("In order to use Google SST/TTS modules, please specify your 'GOOGLE_APPLICATION_CREDENTIALS' in settings.yml")

system.start_system()
print("Dialogue system started!")
