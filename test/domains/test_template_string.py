import pytest

from bn.values.value_factory import ValueFactory
from datastructs.assignment import Assignment
from datastructs.math_expression import MathExpression
from dialogue_system import DialogueSystem
from readers.xml_domain_reader import XMLDomainReader
from settings import Settings
from templates.functional_template import FunctionalTemplate
from templates.template import Template


class TestTemplateString:
    def test_template1(self):
        template = Template.create("this is a first test")
        utterance = "bla bla this is a first test bla"
        assert template.partial_match(utterance).is_matching()

    def test_template2(self):
        template = Template.create("hi my name is {name}")
        utterance1 = "hi my name is Pierre, how are you?"
        assert template.partial_match(utterance1).is_matching()
        utterance2 = "hello how are you?"
        assert not template.partial_match(utterance2).is_matching()
        utterance3 = "hi my name is Pierre"
        assert template.partial_match(utterance3).is_matching()
        assert template.match(utterance3).is_matching()

    def test_template3(self):
        template = Template.create("hi my name is {name} and I need coffee")
        utterance1 = " hi my name is Pierre and i need coffee "
        utterance2 = "hi my name is Pierre and I need coffee right now"
        assert template.partial_match(utterance1).is_matching()
        assert template.partial_match(utterance2).is_matching()
        utterance3 = "hello how are you?"
        assert not template.partial_match(utterance3).is_matching()

        assert not template.match(utterance3).is_matching()
        assert template.match(utterance1).is_matching()

    def test_template4(self):
        template1 = Template.create("hi my name is {name}")
        assert str(template1.match("hi my name is Pierre Lison ").get_value("name")) == "Pierre Lison"

        template2 = Template.create("{name} is my name")
        assert str(template2.match("Pierre Lison is my name").get_value("name")) == "Pierre Lison"

        template3 = Template.create("hi my name is {name} and I need coffee")
        assert str(template3.match("hi my name is Pierre and I need coffee ").get_value("name")) == "Pierre"

    def test_template5(self):
        template1 = Template.create("hi this is {A} and this is {B}")
        assert str(template1.match("hi this is an apple and this is a banana").get_value("A")) == "an apple"
        assert str(template1.match("hi this is an apple and this is a banana").get_value("B")) == "a banana"

    def test_template6(self):
        template1 = Template.create("{anything}")
        assert str(template1.match("bla bla bla").get_value("anything")) == "bla bla bla"

        template2 = Template.create("{anything} is good")
        assert str(template2.match("bla bla bla is good").get_value("anything")) == "bla bla bla"
        assert not template2.match("blo blo").is_matching()
        assert not template2.match("bla bla bla is bad").contains_var("anything")
        assert template2.match("blo is good").is_matching()

        template3 = Template.create("this could be {anything}")
        assert str(template3.match("this could be pretty much anything").get_value("anything")) == "pretty much anything"
        assert not template3.match("but not this").is_matching()
        assert not template3.match("this could beA").is_matching()
        assert not template3.partial_match("this could beA").is_matching()
        assert not template3.match("this could be").is_matching()
        assert not template3.partial_match("this could be").is_matching()

    def test_template7(self):
        template1 = Template.create("here we have slot {A} and slot {B}")
        fillers = Assignment()
        fillers.add_pair("A", "apple")
        fillers.add_pair("B", "banana")
        assert template1.fill_slots(fillers) == "here we have slot apple and slot banana"
        fillers.remove_pair("B")
        assert Template.create(template1.fill_slots(fillers)).get_slots().pop() == "B"

    def test_template8(self):
        template = Template.create("here we have a test")
        assert not template.match("here we have a test2").is_matching()
        assert not template.partial_match("here we have a test2").is_matching()
        assert template.partial_match("here we have a test that is working").is_matching()
        assert not template.match("here we have a test that is working").is_matching()

        template2 = Template.create("bla")
        assert not template2.partial_match("bla2").is_matching()
        assert not template2.partial_match("blabla").is_matching()
        assert template2.partial_match("bla bla").is_matching()
        assert not template2.match("bla bla").is_matching()

    def test_template_quick(self):
        domain = XMLDomainReader.extract_domain("test/data/quicktest.xml")
        system = DialogueSystem(domain)
        system.get_settings().show_gui = False

        system.start_system()
        assert system.get_content("caught").get_prob(False) == pytest.approx(1.0, abs=0.01)
        assert system.get_content("caught2").get_prob(True) == pytest.approx(1.0, abs=0.01)

    def test_template_math(self):
        assert MathExpression("1+2").evaluate() == pytest.approx(3.0, abs=0.001)
        assert MathExpression("-1.2*3").evaluate() == pytest.approx(-3.6, abs=0.001)
        t = Template.create("{X}+2")
        assert str(t.fill_slots(Assignment("X", "3"))) == "5"

    def test_complex_regex(self):
        t = Template.create("a (pizza)? margherita")
        assert t.match("a margherita").is_matching()
        assert t.match("a pizza margherita").is_matching()
        assert not t.match("a pizza").is_matching()
        assert t.partial_match("I would like a margherita").is_matching()

        t2 = Template.create("a (bottle of)? (beer|wine)")
        assert t2.match("a beer").is_matching()
        assert t2.match("a bottle of wine").is_matching()
        assert not t2.match("a bottle of").is_matching()
        assert not t2.match("a coke").is_matching()
        assert t2.partial_match("I would like a bottle of beer").is_matching()

        t3 = Template.create("move (a (little)? bit)? (to the)? left")
        assert t3.match("move a little bit to the left").is_matching()
        assert t3.match("move a bit to the left").is_matching()
        assert t3.match("move to the left").is_matching()
        assert t3.match("move a little bit left").is_matching()
        assert not t3.match("move a to the left").is_matching()

        t4 = Template.create("I want beer(s)?")
        assert t4.match("I want beer").is_matching()
        assert t4.match("I want beers").is_matching()
        assert not t4.match("I want beer s").is_matching()

        t5 = Template.create("(beer(s)?|wine)")
        assert t5.match("beer").is_matching()
        assert t5.match("beers").is_matching()
        assert t5.match("wine").is_matching()
        assert not t5.match("wines").is_matching()
        assert not t5.match("beer wine").is_matching()
        assert Template.create("* (to the|at the)? left of").match("window to the left of").is_matching()
        assert Template.create("* (to the|at the)? left of").match("window left of").is_matching()
        assert Template.create("* (to the|at the)? left of").match("left of").is_matching()

    def test_double(self):
        t = Template.create("MakeOrder({Price})")
        assert t.match("MakeOrder(179)").is_matching()
        assert t.match("MakeOrder(179.0)").is_matching()
        assert not t.match("MakkeOrder(179.0)").is_matching()
        assert not t.match("MakkeOrder()").is_matching()

    def test_match_in_string(self):
        t = Template.create("{X}th of March")
        assert t.match("20th of March").is_matching()
        assert t.partial_match("on the 20th of March").is_matching()
        assert not t.match("20 of March").is_matching()

    def test_star(self):
        t1 = Template.create("here is * test")
        assert t1.match("here is test").is_matching()
        assert t1.match("here is a test").is_matching()
        assert t1.match("here is a great test").is_matching()
        assert not t1.match("here is a bad nest").is_matching()

        t1 = Template.create("* test")
        assert t1.match("test").is_matching()
        assert t1.match("great test").is_matching()
        assert not t1.match("here is a bad nest").is_matching()

        t1 = Template.create("test *")
        assert t1.match("test").is_matching()
        assert t1.match("test that works").is_matching()
        assert not t1.match("nest that is bad").is_matching()

        t1 = Template.create("this is a * {test}")
        assert t1.match("this is a ball").is_matching()
        assert t1.match("this is a really great ball").is_matching()
        assert not t1.match("this is huge").is_matching()
        assert str(t1.match("this is a ball").get_value("test")) == "ball"
        assert str(t1.match("this is a great blue ball").get_value("test")) == "ball"

        t1 = Template.create("* {test}")
        assert str(t1.match("this is a great ball").get_value("test")) == "ball"
        assert str(t1.match("ball").get_value("test")) == "ball"
        t1 = Template.create("{test} *")
        assert str(t1.match("great ball").get_value("test")) == "great ball"
        assert str(t1.match("ball").get_value("test")) == "ball"

    def test_one_char_and_parenthesis(self):
        t = Template.create("?")
        assert t.partial_match("how are you?").is_matching()
        assert t.partial_match("how are you ?").is_matching()
        t = Template.create("Pred1({X})")
        assert t.match("Pred1(FirstTest)").is_matching()
        assert t.match("Pred1(Pred2(Bla))").is_matching()
        assert str(t.match("Pred1(Pred2(Bla))").get_value("X")) == "Pred2(Bla)"
        t = Template.create("Pred2({X},{Y})")
        assert str(t.match("Pred2(Bla,Blo)").get_value("X")) == "Bla"
        assert str(t.match("Pred2(Bla,Blo)").get_value("Y")) == "Blo"
        assert str(t.match("Pred2(Bla(1,2),Blo)").get_value("Y")) == "Blo"
        assert str(t.match("Pred2(Bla,Blo(1,2))").get_value("X")) == "Bla"

    def test_functions(self):
        t = Template.create("{X}+{Y}")
        assert t.fill_slots(Assignment.create_from_string("X=1 ^ Y=2")) == "3"
        assert t.fill_slots(Assignment.create_from_string("X=[1,2] ^ Y=4")) == "[1, 2, 4]"
        t = Template.create("{X}-{Y}")
        assert t.fill_slots(Assignment.create_from_string("X=[1,2] ^ Y=2")) == "[1]"

    def test_real_function(self):
        # def add(*x):
        #     return ValueFactory.create(sum([float(item) for item in x]))
        #
        # def substract(*x):
        #     return ValueFactory.create(value=float(x[0]) - sum([float(item) for item in x[1:]]))
        #
        Settings.add_function("add", lambda *x: ValueFactory.create(sum([float(item) for item in x])))
        Settings.add_function("substract", lambda *x: ValueFactory.create(float(x[0]) - sum([float(item) for item in x[1:]])))

        t = Template.create("add({X},{Y})")

        assert t.fill_slots(Assignment.create_from_string("X=1 ^ Y=2")) == "3"
        t = Template.create("add(4,{Y},{Z})")
        assert t.fill_slots(Assignment.create_from_string("Z=3 ^ Y=2")) == "9"
        t = Template.create("add(4,2)")
        assert t.fill_slots(Assignment.create_from_string("Z=3 ^ Y=2")) == "6"
        t = Template.create("add(substract({X},{Y}),{Z})")
        assert isinstance(t, FunctionalTemplate)
        assert t.fill_slots(Assignment.create_from_string("X=3 ^ Y=1 ^ Z=2")) == "4"
        t = Template.create("add(substract({X},{Y}),substract({Z}, {A}))")
        assert isinstance(t, FunctionalTemplate)
        assert t.fill_slots(Assignment.create_from_string("X=3 ^ Y=1 ^ Z=4 ^ A=2")) == "4"
