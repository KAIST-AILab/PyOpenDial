from example_domains.negotiation.generate_model import GenerateModel
from argparse import Namespace
from parse import parse
import numpy as np
import random
def ste(values):
    return 2 * np.std(values) / np.sqrt(len(values))

args = Namespace(eps=0.0, rl_lr=0.2, momentum=0.0, nesterov=False, visual=False, domain='object_division',
                 temperature=1, num_types=3, num_objects=6, max_score=10, score_threshold=6, seed=1,
                 smart_ai=False, ai_starts=False)

while True:
    product_count = [random.randint(0, 3) for i in range(3)]
    system_value = [random.randint(0, 3) for i in range(3)]
    user_value = [random.randint(0, 3) for i in range(3)]

    # ctx = [count, value, count, value, count, value]
   
    if 10 < product_count[0] * system_value[0] + product_count[1] * system_value[1] + product_count[2] * system_value[2] < 15:
        system_ctx = []
        user_ctx = []

        for i in range(3):
            system_ctx.append(str(product_count[i]))
            system_ctx.append(str(system_value[i]))
            user_ctx.append(str(product_count[i]))
            user_ctx.append(str(user_value[i]))
        break

system_model = GenerateModel(args, system_ctx, 'system')
user_model = GenerateModel(args, user_ctx, 'user')

dialog_lengths = []
rewards = []
agreeds = []
successes = []
for experiment_i in range(1000):
    print("\n\n============== Experiment %d ==============" % experiment_i)
    system_model.agent.feed_context(system_ctx)
    user_model.agent.feed_context(user_ctx)

    dialog_length = 1
    sys_sentence = "hello"
    user_model.read(sys_sentence + "<eos>")
    print('[system]\t%s' % sys_sentence)
    while True:
        dialog_length += 1
        user_sentence, _, _, _ = user_model.write(update=True)
        system_model.read(user_sentence + "<eos>")
        print('[user]\t%s' % user_sentence)
        if user_sentence == '[selection]':
            break
        sys_sentence, _, _, _ = system_model.write(update=True)
        user_model.read(sys_sentence + "<eos>")
        print('[system]\t%s' % sys_sentence)
        if sys_sentence == '[selection]':
            break

    # Selection...
    user_action, _ = user_model.selection(user_model, user_ctx)
    system_action, reward = system_model.selection(system_model, system_ctx)

    system_action = parse('item0={} item1={} item2={}', ' '.join(system_action))
    system_action = [int(x) for x in system_action]
    user_action = parse('item0={} item1={} item2={}', ' '.join(user_action))
    user_action = [int(x) for x in user_action]
    system_context = [int(x) for x in system_ctx]
    system_ctx_count = [system_context[0], system_context[2], system_context[4]]
    system_ctx_value = [system_context[1], system_context[3], system_context[5]]
    success = (system_action[0] + user_action[0]) <= system_ctx_count[0] and \
                  (system_action[1] + user_action[1]) <= system_ctx_count[1] and \
                  (system_action[2] + user_action[2]) <= system_ctx_count[2]


    agreed = (system_action[0] < 100 and system_action[1] < 100 and system_action[2] < 100 and user_action[0] < 100 and user_action[1] < 100 and user_action[2] < 100)
    if success:
        reward = system_action[0] * system_ctx_value[0] + \
                 system_action[1] * system_ctx_value[1] + \
                 system_action[2] * system_ctx_value[2]
    else:
        reward = 0

    successes.append(success)
    agreeds.append(agreed)
    rewards.append(reward)
    dialog_lengths.append(dialog_length)

    print('--------------------------')
    print('Reward     : %6.3f (%6.3f +- %6.3f)' % (reward, np.mean(rewards), ste(rewards)))
    print('Agreed     : %6s (%6.3f +- %6.3f)' % (agreed, np.mean(agreeds), ste(agreeds)))
    print('Success    : %6s (%6.3f +- %6.3f)' % (success, np.mean(successes), ste(successes)))
    print('Length     : %6d (%6.3f +- %6.3f)' % (dialog_length, np.mean(dialog_lengths), ste(dialog_lengths)))
