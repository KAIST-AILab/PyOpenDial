from bn.values.value_factory import ValueFactory
from datastructs.assignment import Assignment
from dialogue_system import DialogueSystem
from readers.xml_domain_reader import XMLDomainReader
from templates.relational_template import RelationalTemplate


def word_cnt_func(*value):
    if len(value) != 1:
        raise ValueError()

    str_val = value[0]
    return ValueFactory.create([len(str_val), len(str_val.split(' '))])


class TestRelation:
    def test_relational(self):
        rel = ValueFactory.create("[sees|tag:VB subject>John object>Anne instrument>[telescope|tag:NN colour>red|tag:ADJ]]")
        assert len(rel) == 5
        assert ValueFactory.create("telescope") in rel.get_sub_values()
        assert str(rel.get_nodes()[0].get_content()) == "sees"

        t = RelationalTemplate("[sees subject>John]")
        assert len(t.get_matches(rel)) == 1

        t = RelationalTemplate("[sees {S}>John]")
        assert len(t.get_matches(rel)) == 1
        assert str(t.get_matches(rel)[0].get_value("S")) == "subject"

        t = RelationalTemplate("[sees {S}>{O}]")
        assert len(t.get_matches(rel)) == 3
        assert str(t.get_matches(rel)[0].get_value("S")) == "instrument"
        assert str(t.get_matches(rel)[0].get_value("O")) == "telescope"

        t = RelationalTemplate("[{V}|tag:{T} subject>{X} object>{Y}]")
        assert str(t.get_matches(rel)[0].get_value("V")) == "sees"
        assert str(t.get_matches(rel)[0].get_value("T")) == "VB"
        assert str(t.get_matches(rel)[0].get_value("X")) == "John"
        assert str(t.get_matches(rel)[0].get_value("Y")) == "Anne"

        t = RelationalTemplate("[sees +>red|tag:{X}]")
        assert len(t.get_matches(rel)) == 1
        assert str(t.get_matches(rel)[0].get_value("X")) == "ADJ"

        rel2 = ValueFactory.create("[sees|tag:VB object>Anne instrument>[telescope|tag:NN colour>red|tag:ADJ] subject>John]")
        assert rel2 == rel
        assert hash(rel2) == hash(rel)
        assert ValueFactory.create("Anne") in rel2

        t = RelationalTemplate("[sees {S}>John]")
        assert len(t.get_slots()) == 1
        assert t.fill_slots(Assignment("S", "subject")) == "[sees subject>John]"

    def test_function(self):
        d = XMLDomainReader.extract_domain("test/data/relationaltest.xml")
        system = DialogueSystem(d)
        system.get_settings().show_gui = False
        system.start_system()
