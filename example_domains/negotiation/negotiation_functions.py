from example_domains.negotiation.negotiation_state import NegotiationState


def generate_user_response(negotiation_state, u_m):
    negotiation_state.user_model.read(u_m + '<eos>')  # update internal state
    sentence, _, _, _ = negotiation_state.user_model.write(update=True)
    negotiation_state.system_model.read(sentence + '<eos>')

    return sentence


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


system_utterance = None


def generate_initial_system_utterance(negotiation_state):
    global system_utterance
    if system_utterance is None:
        system_utterance, _, _, _ = negotiation_state.system_model.write(update=True)

    return system_utterance


def generate_system_option(negotiation_state):
    choice, _ = negotiation_state.system_model.selection(negotiation_state.system_model, negotiation_state.system_ctx)
    return choice


def negotiation_initial_turn():
    negotiation_state = NegotiationState()
    return negotiation_state.initial_turn


def generate_user_selection(negotiation_state):
    choice, _ = negotiation_state.user_model.selection(negotiation_state.user_model, negotiation_state.user_ctx)
    return choice
