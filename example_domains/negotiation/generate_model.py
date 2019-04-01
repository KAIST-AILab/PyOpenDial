from torch.autograd import Variable

from example_domains.negotiation import util
from example_domains.negotiation.agent import RlAgent
import os
import sys

class GenerateModel(object):

    def __init__(self, args, ctx, agent_type):
        model_dir = '%s/example_domains/negotiation' % os.getcwd()
        sys.path.append(model_dir)
        self.model_file = '%s/sv_model.th' % model_dir
        self.model = util.load_model(self.model_file)

        self.agent_type = agent_type
        # self.agent = LstmAgent(self.model, args, name=agent_type)
        # self.agent = LstmRolloutAgent(self.model, args, name=agent_type)
        self.agent = RlAgent(self.model, args, name=agent_type)

        self.ctx = ctx
        self.agent.feed_context(ctx)

    def read(self, input_text):
        """
        Return the response for the given input text
        :param input_text: the given input text
        :return: the generated text output
        """

        if input_text == '':
            # inpt = Variable(self.agent.model.zero_hid(1))
            inpt = ['']
        else:
            inpt = input_text.split() + ['<eos>']
        inpt = Variable(self.agent._encode(inpt, self.agent.model.word_dict))

        lang_hs, self.agent.lang_h = self.agent.model.read(inpt, self.agent.lang_h, self.agent.ctx_h)
        self.agent.lang_hs.append(lang_hs.squeeze(1))
        self.agent.words.append(self.model.word2var('THEM:'))
        self.agent.words.append(inpt)

    def write(self, update=False):
        _, outs, lang_h, lang_hs = self.agent.model.write(self.agent.lang_h, self.agent.ctx_h, 100, self.agent.args.temperature)
        if update:
            self.agent.lang_h = lang_h
            self.agent.lang_hs.append(lang_hs)
            self.agent.words.append(self.model.word2var('YOU:'))
            self.agent.words.append(outs)
        lang_hs_diff = [lang_hs]
        words_diff = [self.model.word2var('YOU:'), outs]

        sentence = ' '.join(self.agent._decode(outs, self.agent.model.word_dict)).replace('<', '[').replace('>', ']').replace(' [eos]', '').strip()
        if sentence == '[selection]':
            sentence = 'selection: How many books, hats, balls do you want to take? (ex. book:1,hat:0,ball:2)'
        return sentence, lang_h, lang_hs_diff, words_diff

    # def generate_response_by_hidden(self, hidden):
    #
    #     _, outs, _, _ = self.agent.model.write(hidden, self.agent.ctx_h, 100, self.agent.args.temperature)
    #
    #     return self.agent._decode(outs, self.agent.model.word_dict)

    def generate_action_set(self, action_num):
        """
        Return the action set for the input text
        :param input_text: the given input text
        :return: the generated action set for the given input text
        """

        # action_list = random.sample(sentences, action_num-1)
        # action_list.append('<selection>')
        action_list = []
        for i in range(action_num):
            sentence, lang_h, lang_hs, words = self.write(update=False)
            action_list.append((sentence, lang_h, lang_hs, words))

        return action_list

    # Previous version: selection with both models
    # def selection(self, models, ctxs):
    #     choices = [][selection]
    #
    #     for model in models:
    #         choice = model.agent.choose()
    #         choice = choice[:model.agent.domain.selection_length() // 2]
    #         choices.append(choice)
    #     agree, rewards = model.agent.domain.score_choices(choices, ctxs)
    #
    #     return choices, agree, rewards

    def selection(self, model, ctx):
        choice = model.agent.choose()
        choice = choice[:model.agent.domain.selection_length() // 2]

        agree, reward = model.agent.domain.score_choices([choice], [ctx])

        result = []
        for i in range(3):
            if choice[i] == "<no_agreement>":
                result.append("item%d:100" % i)
            else:
                result.append(choice[i])

        for i in range(3):
            result[i] = result[i].replace('item0', 'book').replace('item1', 'hat').replace('item2', 'ball').replace('=', ':')

        return ','.join(result), reward[0]

    # def selection_one(self, model, ctx):

        # # evaluate the choices, produce agreement and a reward
        # agree, rewards = self.domain.score_choices(choices, ctxs)
        # logger.dump('-' * 80)
        # logger.dump_agreement(agree)
        # # perform update, in case if any of the agents is learnable
        # for agent, reward in zip(self.agents, rewards):
        #     logger.dump_reward(agent.name, agree, reward)
        #     agent.update(agree, reward)
