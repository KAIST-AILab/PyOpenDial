from bn.b_network import BNetwork
from bn.distribs.distribution_builder import CategoricalTableBuilder as CategoricalTableBuilder, ConditionalTableBuilder as ConditionalTableBuilder
from bn.distribs.continuous_distribution import ContinuousDistribution
from bn.distribs.density_functions.uniform_density_function import UniformDensityFunction
from bn.nodes.chance_node import ChanceNode
from bn.values.value_factory import ValueFactory

from bn.nodes.action_node import ActionNode
from bn.nodes.utility_node import UtilityNode
from datastructs.assignment import Assignment


class NetworkExamples():
    @staticmethod
    def construct_basic_network():
        network = BNetwork()

        builder = CategoricalTableBuilder('Burglary')
        builder.add_row(ValueFactory.create(True), 0.001)
        builder.add_row(ValueFactory.create(False), 0.999)
        b = ChanceNode("Burglary", builder.build())
        network.add_node(b)

        builder = CategoricalTableBuilder('Earthquake')
        builder.add_row(ValueFactory.create(True), 0.002)
        builder.add_row(ValueFactory.create(False), 0.998)
        e = ChanceNode("Earthquake", builder.build())
        network.add_node(e)

        builder = ConditionalTableBuilder('Alarm')
        builder.add_row(Assignment(["Burglary", "Earthquake"]), ValueFactory.create(True), 0.95)
        builder.add_row(Assignment(["Burglary", "Earthquake"]), ValueFactory.create(False), 0.05)
        builder.add_row(Assignment(["Burglary", "!Earthquake"]), ValueFactory.create(True), 0.95)
        builder.add_row(Assignment(["Burglary", "!Earthquake"]), ValueFactory.create(False), 0.05)
        builder.add_row(Assignment(["!Burglary", "Earthquake"]), ValueFactory.create(True), 0.29)
        builder.add_row(Assignment(["!Burglary", "Earthquake"]), ValueFactory.create(False), 0.71)
        builder.add_row(Assignment(["!Burglary", "!Earthquake"]), ValueFactory.create(True), 0.001)
        builder.add_row(Assignment(["!Burglary", "!Earthquake"]), ValueFactory.create(False), 0.999)
        a = ChanceNode("Alarm", builder.build())
        a.add_input_node(b)
        a.add_input_node(e)
        network.add_node(a)

        builder = ConditionalTableBuilder("MaryCalls")
        builder.add_row(Assignment("Alarm"), ValueFactory.create(True), 0.7)
        builder.add_row(Assignment("Alarm"), ValueFactory.create(False), 0.3)
        builder.add_row(Assignment("!Alarm"), ValueFactory.create(True), 0.01)
        builder.add_row(Assignment("!Alarm"), ValueFactory.create(False), 0.99)

        mc = ChanceNode("MaryCalls", builder.build())
        mc.add_input_node(a)
        network.add_node(mc)

        builder = ConditionalTableBuilder("JohnCalls")
        builder.add_row(Assignment(["Alarm"]), ValueFactory.create(True), 0.9)
        builder.add_row(Assignment(["Alarm"]), ValueFactory.create(False), 0.1)
        builder.add_row(Assignment(["!Alarm"]), ValueFactory.create(True), 0.05)
        builder.add_row(Assignment(["!Alarm"]), ValueFactory.create(False), 0.95)
        jc = ChanceNode("JohnCalls", builder.build())
        jc.add_input_node(a)
        network.add_node(jc)

        action = ActionNode("Action")
        action.add_value(ValueFactory.create("CallPolice"))
        action.add_value(ValueFactory.create("DoNothing"))
        network.add_node(action)

        value = UtilityNode("Util1")
        value.add_input_node(b)
        value.add_input_node(action)
        value.add_utility(Assignment(Assignment("Burglary", True), "Action", ValueFactory.create("CallPolice")), -0.5)
        value.add_utility(Assignment(Assignment("Burglary", False), "Action", ValueFactory.create("CallPolice")), -1.0)
        value.add_utility(Assignment(Assignment("Burglary", True), "Action", ValueFactory.create("DoNothing")), 0.0)
        value.add_utility(Assignment(Assignment("Burglary", False), "Action", ValueFactory.create("DoNothing")), 0.0)
        network.add_node(value)

        value2 = UtilityNode("Util2")
        value2.add_input_node(b)
        value2.add_input_node(action)
        value2.add_utility(Assignment(Assignment("Burglary", True), "Action", ValueFactory.create("CallPolice")), 0.0)
        value2.add_utility(Assignment(Assignment("Burglary", False), "Action", ValueFactory.create("CallPolice")), 0.0)
        value2.add_utility(Assignment(Assignment("Burglary", True), "Action", ValueFactory.create("DoNothing")), -10.0)
        value2.add_utility(Assignment(Assignment("Burglary", False), "Action", ValueFactory.create("DoNothing")), 0.5)
        network.add_node(value2)

        return network

    @staticmethod
    def construct_basic_network2():
        network = NetworkExamples.construct_basic_network()
        builder = CategoricalTableBuilder("Burglary")
        builder.add_row(ValueFactory.create(True), 0.1)
        builder.add_row(ValueFactory.create(False), 0.9)
        network.get_chance_node("Burglary").set_distrib(builder.build())
        builder = CategoricalTableBuilder("Earthquake")
        builder.add_row(ValueFactory.create(True), 0.2)
        builder.add_row(ValueFactory.create(False), 0.8)
        network.get_chance_node("Earthquake").set_distrib(builder.build())

        return network

    @staticmethod
    def construct_basic_network3():
        network = NetworkExamples.construct_basic_network()
        network.remove_node("Action")
        ddn = ActionNode("Action")
        network.get_utility_node("Util1").add_input_node(ddn)
        network.get_utility_node("Util1").remove_utility(Assignment(Assignment("Burglary", False), Assignment("Action", "DoNothing")))
        network.get_utility_node("Util2").add_input_node(ddn)
        network.add_node(ddn)
        return network

    @staticmethod
    def construct_basic_network4():
        network = NetworkExamples.construct_basic_network()
        node = ChanceNode("gaussian", ContinuousDistribution("gaussian", UniformDensityFunction(-2, 3)))
        network.add_node(node)
        return network

    @staticmethod
    def construct_iwsds_network():
        network = BNetwork()
        builder = CategoricalTableBuilder("i_u")
        builder.add_row(ValueFactory.create("ki"), 0.4)
        builder.add_row(ValueFactory.create("of"), 0.3)
        builder.add_row(ValueFactory.create("co"), 0.3)
        i_u = ChanceNode("i_u", builder.build())
        network.add_node(i_u)

        builder = ConditionalTableBuilder("a_u")
        builder.add_row(Assignment("i_u", "ki"), ValueFactory.create("ki"), 0.9)
        builder.add_row(Assignment("i_u", "ki"), ValueFactory.create("null"), 0.1)
        builder.add_row(Assignment("i_u", "of"), ValueFactory.create("of"), 0.9)
        builder.add_row(Assignment("i_u", "of"), ValueFactory.create("null"), 0.1)
        builder.add_row(Assignment("i_u", "co"), ValueFactory.create("co"), 0.9)
        builder.add_row(Assignment("i_u", "co"), ValueFactory.create("null"), 0.1)
        a_u = ChanceNode("a_u", builder.build())
        a_u.add_input_node(i_u)
        network.add_node(a_u)

        builder = ConditionalTableBuilder("a_u")
        builder.add_row(Assignment("a_u", "ki"), ValueFactory.create("True"), 0.0)
        builder.add_row(Assignment("a_u", "ki"), ValueFactory.create("False"), 1.0)
        builder.add_row(Assignment("a_u", "of"), ValueFactory.create("True"), 0.6)
        builder.add_row(Assignment("a_u", "of"), ValueFactory.create("False"), 0.4)
        builder.add_row(Assignment("a_u", "co"), ValueFactory.create("True"), 0.15)
        builder.add_row(Assignment("a_u", "co"), ValueFactory.create("False"), 0.85)
        builder.add_row(Assignment("a_u", "null"), ValueFactory.create("True"), 0.25)
        builder.add_row(Assignment("a_u", "null"), ValueFactory.create("False"), 0.75)
        o = ChanceNode("o", builder.build())
        o.add_input_node(a_u)
        network.add_node(o)

        a_m = ActionNode("a_m")
        a_m.add_value(ValueFactory.create("ki"))
        a_m.add_value(ValueFactory.create("of"))
        a_m.add_value(ValueFactory.create("co"))
        a_m.add_value(ValueFactory.create("rep"))
        network.add_node(a_m)

        r = UtilityNode("r")
        r.add_input_node(a_m)
        r.add_input_node(i_u)
        r.add_utility(Assignment(Assignment("a_m", "ki"), Assignment("i_u", "ki")), 3)
        r.add_utility(Assignment(Assignment("a_m", "ki"), Assignment("i_u", "of")), -5)
        r.add_utility(Assignment(Assignment("a_m", "ki"), Assignment("i_u", "co")), -5)

        r.add_utility(Assignment(Assignment("a_m", "of"), Assignment("i_u", "ki")), -5)
        r.add_utility(Assignment(Assignment("a_m", "of"), Assignment("i_u", "of")), 3)
        r.add_utility(Assignment(Assignment("a_m", "of"), Assignment("i_u", "co")), -5)

        r.add_utility(Assignment(Assignment("a_m", "co"), Assignment("i_u", "ki")), -5)
        r.add_utility(Assignment(Assignment("a_m", "co"), Assignment("i_u", "of")), -5)
        r.add_utility(Assignment(Assignment("a_m", "co"), Assignment("i_u", "co")), 3)

        r.add_utility(Assignment(Assignment("a_m", "rep"), Assignment("i_u", "ki")), -0.5)
        r.add_utility(Assignment(Assignment("a_m", "rep"), Assignment("i_u", "of")), -0.5)
        r.add_utility(Assignment(Assignment("a_m", "rep"), Assignment("i_u", "co")), -0.5)
        network.add_node(r)

        return network
