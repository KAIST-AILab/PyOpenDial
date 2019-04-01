from collections import Callable

from multipledispatch import dispatch


class CustomUtilityFunction:

    def __init__(self, action_node_id, simulation_action_node_id, func):
        if isinstance(action_node_id, str) and isinstance(func, Callable):
            if not isinstance(action_node_id, str) or not isinstance(func, Callable):
                raise NotImplementedError("UNDEFINED PARAMETERS")

            self._action_node_id = action_node_id
            self._simulation_action_node_id = simulation_action_node_id
            self._func = func
        else:
            raise NotImplementedError()

    @dispatch(object)  # object: DialogueState
    def query_util(self, state):
        return self._func(self._simulation_action_node_id, state)

    @dispatch()
    def get_action_node_id(self):
        return self._action_node_id

    @dispatch()
    def get_simulation_action_node_id(self):
        return self._simulation_action_node_id
