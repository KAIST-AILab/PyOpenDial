import subprocess
import numpy as np
from time import time
from parse import parse

def ste(values):
    return 2 * np.std(values) / np.sqrt(len(values))

##################################
# Experimental Settings
planner = 'forward'  # 'forward' or 'mcts'
##################################

rewards = []
dialog_lengths = []
time_per_steps = []
agreeds = []
successes = []
for experiment_i in range(1000):
    start_time = time()
    print("\n\n======== Experiment %d (planner=%s) ========" % (experiment_i, planner))
    p = subprocess.Popen('python dialogue_system.py --domain example_domains/negotiation/negotiation.xml --planner %s' % planner, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log = []
    dialog_length = 0
    terminal = False
    lines = p.stdout.readlines()
    lines = [line.decode('ascii').strip() for line in lines]
    # print('\n'.join(lines))
    for line in lines:
        if '[user]' in line or '[system]' in line:
            log.append(line)
            if not terminal:
                dialog_length += 1
            if '[selection]' in line:
                terminal = True

    try:
        elapsed_time = time() - start_time

        if int((len(log) - 1) / 2) == 0:
            time_per_step = elapsed_time / int((len(log)) / 2)
        else:
            time_per_step = elapsed_time / int((len(log) - 1) / 2)
        # Parse final result
        parse_result = parse('[system]\t(system action [{}], user action [{}], system context {}, reward {})', log[-1])
        # print (parse_result[0])
        system_action = parse('item0={} item1={} item2={}', parse_result[0])
        system_action = [int(x) for x in system_action]
        user_action = parse('item0={} item1={} item2={}', parse_result[1])
        user_action = [int(x) for x in user_action]
        system_context = parse("['{}', '{}', '{}', '{}', '{}', '{}']", parse_result[2])
        system_context = [int(x) for x in system_context]
        system_ctx_count = [system_context[0], system_context[2], system_context[4]]
        system_ctx_value = [system_context[1], system_context[3], system_context[5]]
        reward = float(parse_result[3])
        success = (system_action[0] + user_action[0]) <= system_ctx_count[0] and \
                  (system_action[1] + user_action[1]) <= system_ctx_count[1] and \
                  (system_action[2] + user_action[2]) <= system_ctx_count[2]

        agreed = (system_action[0] < 100 and system_action[1] < 100 and system_action[2] < 100 and user_action[0] < 100 and user_action[1] < 100 and user_action[2] < 100)
        ##############
        dialog_lengths.append(dialog_length)
        rewards.append(reward)
        time_per_steps.append(time_per_step)
        successes.append(success)
        agreeds.append(agreed)

        print('\n'.join(log))
        print('---------------------------------')
        print("Context count     : %s" % system_ctx_count)
        print("System's ctx value: %s" % system_ctx_value)
        print("System's selection: %s" % system_action)
        print("User's selection  : %s" % user_action)
        print('Reward     : %6.3f (%6.3f +- %6.3f)' % (reward, np.mean(rewards), ste(rewards)) )
        print('Length     : %6d (%6.3f +- %6.3f)' % (dialog_length, np.mean(dialog_lengths), ste(dialog_lengths)) )
        print('TimePerStep: %6.3f (%6.3f +- %6.3f) / system said %d times' % (time_per_step, np.mean(time_per_steps), ste(time_per_steps), int((len(log) - 1) / 2) ))
        print('Agreed     : %6s (%6.3f +- %6.3f)' % (agreed, np.mean(agreeds), ste(agreeds)))
        print('Success    : %6s (%6.3f +- %6.3f)' % (success, np.mean(successes), ste(successes)))
    except:
        print('Error occured...')
        print('\n'.join(log))

dialog_lengths = np.array(dialog_lengths)
print('=================== Result =================')
print('Length: %f +- %f' % ( np.mean(dialog_lengths), ste(dialog_lengths) ))
print('Reward: %f +- %f' % ( np.mean(rewards), ste(rewards) ))
print('TimePerStep: %f +- %f' % ( np.mean(time_per_steps), ste(time_per_steps) ))
print('Agreed: %f +- %f' % ( np.mean(agreeds), ste(agreeds) ))
print('Successes: %f +- %f' % ( np.mean(successes), ste(successes) ))
