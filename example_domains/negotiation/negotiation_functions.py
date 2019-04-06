from parse import parse

from example_domains.negotiation.negotiation_state import NegotiationState

system_utterance = dict()
system_selection = dict()
user_utterance = dict()
user_selection = dict()


def negotiation_initial_turn():
    negotiation_state = NegotiationState()
    return negotiation_state.initial_turn


def initial_message():
    def plural(n):
        if n <= 1:
            return ""
        return "s"
    negotiation_state = NegotiationState()
    book_cnt, book_val, hat_cnt, hat_val, ball_cnt, ball_val = [int(x) for x in negotiation_state.user_ctx]

    message = "There are %d book%s, %d hat%s, and %d ball%s on the table." % (
        book_cnt, plural(book_cnt), hat_cnt, plural(hat_cnt), ball_cnt, plural(ball_cnt)
    )
    message += "\nFor you, each book has a value of %d, each hat has a value of %d, and each ball has a value of %d" % (
        book_val, hat_val, ball_val
    )
    message += "\n========================================================"
    message += "\nNow, let's start the negotiation."

    if negotiation_state.initial_turn == 'system':
        message += " This time, I will start first."
    else:
        message += " This time, you will start first.\n(ex. say: 'Give me a hat. Then I will give you a ball.')"

    return message


def update_dialogue_history(dialogue_history, turn_num, turn, utterance):
    """
    :param dialogue_history: the dialogue history
    :param turn: 'user' or 'system'
    :param utterance: the utterance
    """
    turn_num = int(turn_num)
    utterances = dialogue_history.split("\n")
    utterances.remove('')
    if len(utterances) > 0 and utterances[-1] == "<%s>%s" % (turn, utterance):
        return dialogue_history
    elif 'selection' in utterance:
        return dialogue_history
    elif 'book:' in utterance and 'hat:' in utterance and 'ball:' in utterance:
        return dialogue_history
    elif 'There are' in utterance and 'For you, each book has a value of' in utterance:
        return dialogue_history
    else:
        return dialogue_history + "\n<%s>%s" % (turn, utterance)


def update_negotiation_agent(negotiation_state, dialogue_history, is_user):
    negotiation_agent = negotiation_state.user_agent if is_user else negotiation_state.system_agent
    # reset agent
    if is_user:
        negotiation_agent.feed_context(negotiation_state.user_ctx)
    else:
        negotiation_agent.feed_context(negotiation_state.system_ctx)

    # read dialogue history
    utterances = dialogue_history.split("\n")
    utterances.remove('')
    for utterance in utterances:
        if '<user>' in utterance:
            prefix_token = 'YOU:' if is_user else 'THEM:'
        elif '<system>' in utterance:
            prefix_token = 'THEM:' if is_user else 'YOU:'
        negotiation_agent.read(utterance.replace('<user>', '').replace('<system>', '') + ' <eos>', prefix_token=prefix_token)

    return len(utterances)


def generate_system_utterance(negotiation_state, dialogue_history, turn_num, u_u, idx):
    if u_u:
        dialogue_history = update_dialogue_history(dialogue_history, turn_num, 'user', u_u)
    idx = int(idx)

    global system_utterance
    if (dialogue_history, idx) not in system_utterance:
        # update system agent
        update_negotiation_agent(negotiation_state, dialogue_history, is_user=False)
        system_utterance[(dialogue_history, idx)] = negotiation_state.system_agent.write()

    return system_utterance[(dialogue_history, idx)]


def generate_system_selection(negotiation_state, dialogue_history, turn_num, u_u):
    if u_u:
        dialogue_history = update_dialogue_history(dialogue_history, turn_num, 'user', u_u)
    global system_selection
    if dialogue_history not in system_selection:
        update_negotiation_agent(negotiation_state, dialogue_history, is_user=False)
        system_selection[dialogue_history] = negotiation_state.system_agent.choose()

    return system_selection[dialogue_history]


def generate_user_utterance(negotiation_state, dialogue_history, turn_num, u_m):
    if u_m:
        dialogue_history = update_dialogue_history(dialogue_history, turn_num, 'system', u_m)
    global user_utterance
    if dialogue_history not in user_utterance:
        update_negotiation_agent(negotiation_state, dialogue_history, is_user=True)
        user_utterance[dialogue_history] = negotiation_state.user_agent.write()

    # print('GENERATE USER UTTERANCE')
    # print(dialogue_history.split("\n"))
    # print(user_utterance[dialogue_history])
    # print()

    return user_utterance[dialogue_history]


def generate_user_selection(negotiation_state, dialogue_history):
    global user_selection
    if dialogue_history not in user_selection:
        cnt = update_negotiation_agent(negotiation_state, dialogue_history, is_user=True)
        if cnt > 0:
            user_selection[dialogue_history] = negotiation_state.user_agent.choose()
        else:
            user_selection[dialogue_history] = "book=100,hat=100,ball=100"

    # print('GENERATE USER SELECTION')
    # print(dialogue_history.split("\n"))
    # print(user_selection[dialogue_history])
    # print()

    return user_selection[dialogue_history]


def show_result(negotiation_state, dialogue_history, turn_num, u_u):
    system_selection_utterance = generate_system_selection(negotiation_state, dialogue_history, turn_num, u_u)
    system_selection = [int(x) for x in parse('book:{},hat:{},ball:{}', system_selection_utterance)]
    reward = compute_user_reward(negotiation_state, dialogue_history, turn_num, u_u)
    return 'System selected %d books, %d hats, and %d balls. Your reward is %d.' % (system_selection[0], system_selection[1], system_selection[2], reward)


def compute_user_reward(negotiation_state, dialogue_history, turn_num, u_u):
    system_selection_utterance = generate_system_selection(negotiation_state, dialogue_history, turn_num, u_u)
    user_selection_utterance = u_u

    system_selection = parse('book:{},hat:{},ball:{}', system_selection_utterance)
    user_selection = parse('book:{},hat:{},ball:{}', user_selection_utterance)
    try:
        reward = 0.
        for i in range(3):
            if int(system_selection[i]) + int(user_selection[i]) > int(negotiation_state.user_ctx[i * 2]):
                return 1e-3
            reward += int(user_selection[i]) * int(negotiation_state.user_ctx[i * 2 + 1])
        return float(reward)
    except:
        return 0.


def compute_system_reward(negotiation_state, dialogue_history, turn_num, u_u):
    system_selection_utterance = generate_system_selection(negotiation_state, dialogue_history, turn_num, u_u)
    user_selection_utterance = u_u

    system_selection = parse('book:{},hat:{},ball:{}', system_selection_utterance)
    user_selection = parse('book:{},hat:{},ball:{}', user_selection_utterance)
    try:
        reward = 0.001
        for i in range(3):
            if int(system_selection[i]) + int(user_selection[i]) > int(negotiation_state.system_ctx[i * 2]):
                return 0.001
            reward += int(system_selection[i]) * int(negotiation_state.system_ctx[i * 2 + 1])

        return float(reward)
    except:
        return 0.001


def increase_turn_num(n):
    return int(n) + 1
