from bn.values.value_factory import ValueFactory
from datastructs.assignment import Assignment


class TestAssignment:
    def test_assign_interchance(self):
        a1 = Assignment(Assignment("Burglary", True), "Earthquake", ValueFactory.create(False))
        a1bis = Assignment(Assignment("Earthquake", False), "Burglary", ValueFactory.create(True))
        a2 = Assignment(Assignment("Burglary", False), "Earthquake", ValueFactory.create(True))
        a2bis = Assignment(Assignment("Earthquake", True), "Burglary", ValueFactory.create(False))

        assert a1 != a2
        assert hash(a1) != hash(a2)
        assert a1bis != a2bis
        assert hash(a1bis) != hash(a2bis)
        assert a1 != a2bis
        assert hash(a1) != hash(a2bis)
        assert a1bis != a2
        assert hash(a1bis) != hash(a2)