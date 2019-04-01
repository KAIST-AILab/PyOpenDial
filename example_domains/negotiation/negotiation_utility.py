from parse import parse

from bn.distribs.utility_table import UtilityTable
from datastructs.assignment import Assignment
import numpy as np
from copy import copy


def get_action_info_by_action_row(action_infos, action_assignment):
    """
    :param action_infos: list of (sentence, lang_h, lang_hs, words)
    :param action_assignment: row of utility table
    :return:
    """
    action_sentence = str(action_assignment.get_values()[0])
    if action_infos is not None:
        for info in action_infos:
            sentence, lang_h, lang_hs, words = info
            if sentence == action_sentence:
                return sentence, lang_h, lang_hs, words
    return None, None, None, None


def apply_changes_func(utility_table, state, action):
    """
    :param state: the dialogue state
    :param action: row of utility table (assignment)
    :param utility_table: utility table
    :return:
    """
    action_infos = utility_table.get_additional_info()
    negotiation_state = state.get_chance_node('negotiation_state').sample().get_value()
    system_agent = negotiation_state.system_model.agent
    user_agent = negotiation_state.user_model.agent
    system_sentence, system_lang_h, system_lang_hs, system_words = get_action_info_by_action_row(action_infos, action)

    system_lang_h_bak, system_lang_hs_bak, system_words_bak = \
        copy(system_agent.lang_h), copy(system_agent.lang_hs), copy(system_agent.words)
    user_lang_h_bak, user_lang_hs_bak, user_words_bak = \
        copy(user_agent.lang_h), copy(user_agent.lang_hs), copy(user_agent.words)
    system_agent.lang_h = system_lang_h
    system_agent.lang_hs.extend(system_lang_hs)
    system_agent.words.extend(system_words)
    return (system_lang_h_bak, system_lang_hs_bak, system_words_bak, user_lang_h_bak, user_lang_hs_bak, user_words_bak)


def rollback_changes_func(utility_table, state, action, before_values):
    negotiation_state = state.get_chance_node('negotiation_state').sample().get_value()
    system_agent = negotiation_state.system_model.agent
    user_agent = negotiation_state.user_model.agent

    system_lang_h_bak, system_lang_hs_bak, system_words_bak, user_lang_h_bak, user_lang_hs_bak, user_words_bak = before_values
    system_agent.lang_h = system_lang_h_bak
    system_agent.lang_hs = system_lang_hs_bak
    system_agent.words = system_words_bak
    user_agent.lang_h = user_lang_h_bak
    user_agent.lang_hs = user_lang_hs_bak
    user_agent.words = user_words_bak


def system_utility(simulation_action_node_id, state):
    simulation_action_node_id += "'"
    negotiation_state = state.get_chance_node('negotiation_state').sample().get_value()
    u_u = str(state.get_chance_node('u_u').get_distrib().get_best())
    u_m = str(state.get_chance_node('u_m').get_distrib().get_best())
    current_step = str(state.get_chance_node('current_step').get_distrib().get_best())

    # update internal state
    if len(negotiation_state.user_model.agent.lang_hs) > 0:
        negotiation_state.user_model.read(u_m + '<eos>')
    negotiation_state.user_model.read(u_u + '<eos>')
    negotiation_state.system_model.read(u_u + '<eos>')

    utility_table = UtilityTable()
    if current_step == 'Negotiation' and 'selection' in u_u:
        # selection
        utility_table.set_util(Assignment('current_step', 'Result'), 1e-3)
        utility_table.set_util(Assignment(simulation_action_node_id, 'How many books, hats, and balls do you want to take? (ex. book:1,hat:0,ball:2)'), 1e-3)
    elif current_step == 'Negotiation':
        # general negotiation
        action_infos = negotiation_state.system_model.generate_action_set(negotiation_state.action_num)
        for sentence, lang_h, lang_hs, words in action_infos:
            sentence_assignment = Assignment(simulation_action_node_id, sentence)
            utility_table.set_util(Assignment(sentence_assignment), 1e-3)
        utility_table.set_custom_utility(action_infos, apply_changes_func, rollback_changes_func)
    elif current_step == 'Result':
        system_action, _ = negotiation_state.system_model.selection(negotiation_state.system_model, negotiation_state.system_ctx)
        user_action = u_u
        system_selection = [int(x) for x in parse('book:{},hat:{},ball:{}', system_action)]
        reward = compute_reward(negotiation_state.user_ctx, system_action, user_action)
        utility_table.set_util(Assignment('current_step', 'Terminated'), 1e-3)
        utility_table.set_util(Assignment(simulation_action_node_id, 'System selected %d books, %d hats, and %d balls. Your reward is %d'
                                          % (system_selection[0], system_selection[1], system_selection[2], reward)), 1e-3)
    elif current_step == 'Terminated':
        utility_table.set_util(Assignment(simulation_action_node_id, 'Terminated...'), 1e-3)

    return utility_table


def system_utility_simulation(simulation_action_node_id, state):
    simulation_action_node_id += "'"
    negotiation_state = state.get_chance_node('negotiation_state').sample().get_value()
    u_u = str(state.get_chance_node('u_u_sim').get_distrib().get_best())
    u_m = str(state.get_chance_node('u_m_sim').get_distrib().get_best())
    current_step = str(state.get_chance_node('current_step').get_distrib().get_best())

    utility_table = UtilityTable()
    if current_step == 'Negotiation' and ("selection" in u_u or "selection" in u_m):
        # selection: compute reward
        choice_system, _ = negotiation_state.system_model.selection(negotiation_state.system_model, negotiation_state.system_ctx)
        choice_user, _ = negotiation_state.user_model.selection(negotiation_state.user_model, negotiation_state.user_ctx)
        reward = compute_reward(negotiation_state.system_ctx, choice_user, choice_system)
        utility_table.set_util(Assignment('current_step', 'Terminated'), 1e-3)
        utility_table.set_util(Assignment(simulation_action_node_id, 'terminated'), reward)

    elif current_step == 'Negotiation':
        # general negotiation
        action_infos = negotiation_state.system_model.generate_action_set(negotiation_state.action_num)
        for sentence, lang_h, lang_hs, words in action_infos:
            sentence_assignment = Assignment(simulation_action_node_id, sentence)
            utility_table.set_util(Assignment(sentence_assignment), 1e-3)
        utility_table.set_custom_utility(action_infos, apply_changes_func, rollback_changes_func)

    return utility_table


def compute_reward(target_ctx, opponent_action, target_action):
    opponent_action = parse('book:{},hat:{},ball:{}', opponent_action)
    target_action = parse('book:{},hat:{},ball:{}', target_action)

    try:
        reward = 0
        for i in range(3):
            if int(opponent_action[i]) + int(target_action[i]) > int(target_ctx[i * 2]):
                return 1e-3
            reward += int(target_action[i]) * int(target_ctx[i * 2 + 1])
    except:
        raise ValueError()

    return float(reward)
