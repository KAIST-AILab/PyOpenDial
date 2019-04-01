from bn.nodes.custom_utility_function import CustomUtilityFunction
from dialogue_state import DialogueState
from domains.rules.rule import Rule
from templates.template import Template
from collections import Collection

import logging
from multipledispatch import dispatch


class Model:
    """
    Representation of a rule model -- that is, a collection of rules of identical
    types (prediction, decision or update), associated with a trigger on specific
    variables.
    """

    # logger
    log = logging.getLogger('PyOpenDial')

    id_counter = 0

    # ===================================
    # MODEL CONSTRUCTION
    # ===================================

    def __init__(self):
        """
        Creates a new model, with initially no trigger and an empty list of rules
        """
        self._blocking = False
        self._triggers = []
        self._rules = []
        self._custom_utility_function = None
        self._id = "model" + str(Model.id_counter)
        Model.id_counter += 1

    @dispatch()
    def start(self):
        pass

    @dispatch(bool)
    def pause(self, should_be_paused):
        pass

    @dispatch(str)
    def set_id(self, a_id):
        """
        Changes the identifier for the model

        :param a_id: the model identifier
        """
        self._id = a_id

    @dispatch(str)
    def add_trigger(self, trigger):
        """
        Adds a new trigger to the model, defined by the variable label

        :param trigger: the variable
        """
        self._triggers.append(Template.create(trigger))

    @dispatch(list)
    def add_triggers(self, triggers):
        """
        Adds a list of triggers to the model, defined by the variable label

        :param triggers: the list of triggers
        """
        for s in triggers:
            self.add_trigger(s)

    @dispatch(Rule)
    def add_rule(self, rule):
        """
        Adds a new rule to the model

        :param rule: the rule to add
        """
        self._rules.append(rule)

    @dispatch(bool)
    def set_blocking(self, blocking):
        """
        Sets the model as "blocking" (forbids other models to be triggered in parallel
        when this model is triggered). Default is false.

        :param blocking: whether to set the model in blocking mode or not
        """
        self._blocking = blocking

    @dispatch(CustomUtilityFunction)
    def set_custom_utility_function(self, custom_utility_function):
        assert(self._custom_utility_function is None)
        self._custom_utility_function = custom_utility_function

    @dispatch()
    def get_custom_utility_function(self):
        return self._custom_utility_function

    # ===================================
    # GETTERS
    # ===================================

    @dispatch()
    def get_id(self):
        """
        Returns the model identifier

        :return: the model identifier
        """
        return self._id

    @dispatch()
    def get_rules(self):
        """
        Returns the list of rules contained in the model

        :return: the list of rules
        """
        return list(self._rules)

    @dispatch(DialogueState)
    def trigger(self, state):
        """
        Triggers the model with the given state and list of recently updated variables.

        :param state: the current dialogue state
        :return: true if the state has been changed, false otherwise
        """
        for rule in self._rules:
            try:
                state.apply_rule(rule)
            except Exception as e:
                self.log.warning("rule " + rule.get_rule_id() + " could not be applied ")
                raise ValueError()

        if self._custom_utility_function is not None:
            state.apply_custom_utility_function(self._custom_utility_function)

        return len(state.get_new_variables()) > 0 or len(state.get_new_action_variables()) > 0

    # TODO: 버그인지 확인 필요.
    @dispatch(DialogueState, Collection)
    def is_triggered(self, state, updated_vars):
        """
        Returns true if the model is triggered by the updated variables.

        :param state: the dialogue state
        :param updated_vars: the updated variables
        :return: true if triggered, false otherwise
        """
        if updated_vars is None:
            raise ValueError()

        if len(self._rules) == 0 and self._custom_utility_function is None:
            return False
        for trigger in self._triggers:
            for updated_var in updated_vars:
                if trigger.match(updated_var).is_matching():
                    return True
        return False

    @dispatch(Collection)
    def is_triggered(self, updated_vars):
        """
        Returns true if the model is triggered by the updated variables.

        :param updated_vars: updatedVars the updated variables
        :return: true if triggered, false otherwise
        """
        if updated_vars is None:
            raise ValueError()

        if len(self._rules) == 0 and self._custom_utility_function is None:
            return False
        for trigger in self._triggers:
            for updated_var in updated_vars:
                if trigger.match(updated_var).is_matching():
                    return True
        return False

    @dispatch()
    def get_triggers(self):
        """
        Returns the model triggers

        :return: the model triggers
        """
        return self._triggers

    @dispatch()
    def is_blocking(self):
        """
        Returns true if the model is set in "blocking" mode and false otherwise.

        :return: whether blocking mode is activated or not (default is false).
        """
        return self._blocking

    # ===================================
    # UTILITY METHODS
    # ===================================

    def __str__(self):
        """
        Returns the string representation of the model
        """
        res = self._id
        res += " [triggers="
        for trigger in self._triggers:
            res += "(" + str(trigger) + ")" + " v "
        res = res[0:-3] + "] with " + str(len(self._rules)) + " rules: "

        for rule in self._rules:
            res += rule.get_rule_id() + ","
        return res[0:-1]

    def __hash__(self):
        """
        Returns the hashcode for the model

        :return: the hashcode
        """
        return hash(self._id) + hash(self._triggers) - hash(self._rules)
