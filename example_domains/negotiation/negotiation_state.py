import random
from example_domains.negotiation.generate_model import GenerateModel
from utils.py_utils import Singleton


class NegotiationState(Singleton):

    initialized = False

    def __init__(self):
        if NegotiationState.initialized:
            return
        NegotiationState.initialized = True

        from argparse import Namespace
        args = Namespace(eps=0.0, rl_lr=0.2, momentum=0.0, nesterov=False, visual=False, domain='object_division',
                         context_file='data/negotiate/selfplay.txt',
                         temperature=1, num_types=3, num_objects=6, max_score=10, score_threshold=6, seed=1,
                         smart_ai=False, ai_starts=False, ref_text='data/negotiate/train.txt')

        while True:
            product_count = [random.randint(1, 4) for i in range(3)]
            system_value = [random.randint(0, 5) for i in range(3)]
            user_value = [random.randint(0, 5) for i in range(3)]

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

        self.system_ctx = system_ctx
        self.system_model = GenerateModel(args, system_ctx, 'system')

        self.user_ctx = user_ctx
        self.user_model = GenerateModel(args, user_ctx, 'user')

        self.action_num = 3  # used for planning
        if random.randint(0, 1):
            self.initial_turn = "system"
        else:
            self.initial_turn = "user"

    def __copy__(self):
        raise NotImplementedError()
