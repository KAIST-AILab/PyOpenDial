import pytest

from bn.distribs.distribution_builder import CategoricalTableBuilder
from bn.values.array_val import ArrayVal
from bn.values.boolean_val import BooleanVal
from bn.values.double_val import DoubleVal
from bn.values.set_val import SetVal
from bn.values.string_val import StringVal
from bn.values.value_factory import ValueFactory
from datastructs.assignment import Assignment
import numpy as np

class TestValue:
    def test_assign(self):
        a = Assignment.create_from_string('blabla=3 ^ !bloblo^TTT=32.4 ^v=[0.4,0.6] ^ final')

        assert len(a.get_variables()) == 5
        assert a.get_variables() == {'blabla', 'bloblo', 'TTT', 'v', 'final'}
        assert a.get_value('blabla') == ValueFactory.create('3')
        assert a.get_value('bloblo') == ValueFactory.create(False)
        assert a.get_value('TTT') == ValueFactory.create('32.4')
        assert a.get_value('v') == ValueFactory.create([0.4, 0.6])
        assert a.get_value('final') == ValueFactory.create(True)

    def test_classical(self):
        assert isinstance(ValueFactory.create(' blabla '), StringVal)
        assert isinstance(ValueFactory.create('3'), DoubleVal)
        assert isinstance(ValueFactory.create('3.6'), DoubleVal)
        assert ValueFactory.create('3').get_double() == pytest.approx(3.0, abs=0.0001)
        assert isinstance(ValueFactory.create('[firstItem, secondItem, 3.6]'), SetVal)
        assert len(ValueFactory.create('[firstItem, secondItem, 3.6]').get_sub_values()) == 3
        assert ValueFactory.create('[firstItem, secondItem, 3.6]').get_sub_values() == {ValueFactory.create('firstItem'), ValueFactory.create('secondItem'), ValueFactory.create(3.6)}
        assert isinstance(ValueFactory.create('[0.6, 0.4, 32]'), ArrayVal)
        assert len(ValueFactory.create('[0.6, 0.4, 32]').get_array()) == 3
        assert ValueFactory.create('[0.6, 0.4, 32]').get_array()[2] == pytest.approx(32, abs=0.0001)
        assert isinstance(ValueFactory.create('True'), BooleanVal)
        assert not ValueFactory.create('False').get_boolean()
        assert ValueFactory.create('None') == ValueFactory.none()
        assert not ValueFactory.create('firsttest').__lt__(ValueFactory.create('firsttest'))
        assert ValueFactory.create('firsttest').__lt__(ValueFactory.create('secondTest'))
        assert ValueFactory.create(3.0).__lt__(ValueFactory.create(5.0))
        assert not ValueFactory.create(5.0).__lt__(ValueFactory.create(3.0))
        assert (ValueFactory.create(5.0).__lt__(ValueFactory.create('test'))) == (ValueFactory.create('test').__lt__(ValueFactory.create(5.0)))
        assert len(ValueFactory.create('[test,[1,2],True]').get_sub_values()) == 3
        assert ValueFactory.create('test') in ValueFactory.create('[test,[1,2],True]').get_sub_values()
        assert ValueFactory.create('[1,2]') in ValueFactory.create('[test,[1,2],True]').get_sub_values()
        assert ValueFactory.create('True') in ValueFactory.create('[test,[1,2],True]').get_sub_values()
        assert len(ValueFactory.create('[a1=test,a2=[1,2],a3=true]').get_sub_values()) == 3

    def test_closest(self):
        builder = CategoricalTableBuilder('v')
        builder.add_row(np.array([0.2, 0.2]), 0.3)
        builder.add_row(np.array([0.6, 0.6]), 0.4)
        table = builder.build()
        assert table.get_prob(np.array([0.25, 0.3])) == pytest.approx(0.3, abs=0.01)
        assert table.get_prob(np.array([0.5, 0.4])) == pytest.approx(0.4, abs=0.01)
