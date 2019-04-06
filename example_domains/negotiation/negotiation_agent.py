import os
import sys

from torch.autograd import Variable

from example_domains.negotiation import util
from example_domains.negotiation.agent import LstmAgent


class NegotiationAgent(object):

    def __init__(self, args, ctx, agent_type):
        model_dir = '%s/example_domains/negotiation' % os.getcwd()
        sys.path.append(model_dir)
        self.model_file = '%s/sv_model.th' % model_dir
        self.model = util.load_model(self.model_file)

        self.agent_type = agent_type
        self.agent = LstmAgent(self.model, args, name=agent_type)

        self.ctx = ctx
        self.agent.feed_context(ctx)

    def read(self, input_text, prefix_token):
        """
        Return the response for the given input text
        :param input_text: the given input text
        :param prefix_token: 'YOU:' or 'THEM:'
        :return: the generated text output
        """

        if input_text == '':
            # inpt = Variable(self.agent.model.zero_hid(1))
            inpt = ['']
        else:
            inpt = input_text.split() + ['<eos>']
        inpt = Variable(self.agent._encode(inpt, self.agent.model.word_dict))

        lang_hs, self.agent.lang_h = self.agent.model.read(inpt, self.agent.lang_h, self.agent.ctx_h, prefix_token=prefix_token)
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

        sentence = ' '.join(self.agent._decode(outs, self.agent.model.word_dict)).replace('<', '[').replace('>', ']').replace(' [eos]', '').strip()
        return sentence

    def choose(self):
        choice = self.agent.choose()
        choice = choice[:self.agent.domain.selection_length() // 2]

        agree, reward = self.agent.domain.score_choices([choice], [self.ctx])

        result = []
        for i in range(3):
            if choice[i] == "<no_agreement>":
                result.append("item%d:100" % i)
            else:
                result.append(choice[i])

        for i in range(3):
            result[i] = result[i].replace('item0', 'book').replace('item1', 'hat').replace('item2', 'ball').replace('=', ':')

        return ','.join(result)

    def feed_context(self, context):
        self.agent.feed_context(context)
