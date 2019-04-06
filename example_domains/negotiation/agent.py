# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
A set of classes that facilitate a dialogue between agents.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import example_domains.negotiation.domain as domain


class Agent(object):
    """Agent's interface.

    The dialogue should proceed in the following way:

    1) feed_context to each of the agent.
    2) randomly pick an agent who will start the conversation.
    3) the starting agent will write down her utterance.
    4) the other agent will read the pronounced utterance.
    5) unless the end of dialogue is pronounced, swap the agents and repeat the steps 3-4.
    6) once the conversation is over, generate choices for each agent and calculate the reward.
    7) pass back to the reward to the update function.
    """

    def feed_context(self, context):
        """Feed context in to start new conversation.

        context: a list of context tokens.
        """
        pass

    def read(self, inpt):
        """Read an utterance from your partner.

        inpt: a list of English words describing a sentence.
        """
        pass

    def write(self):
        """Generate your own utterance."""
        pass

    def choose(self):
        """Call it after the conversation is over, to make the selection."""
        pass

    def update(self, agree, reward):
        """After end of each dialogue the reward will be passed back to update the parameters.

        agree: a boolean flag that specifies if the agents agreed on the deal.
        reward: the reward that the agent receives after the dialogue. 0 if there is no agreement.
        """
        pass


class LstmAgent(Agent):
    """An agent that uses DialogModel as an AI."""
    def __init__(self, model, args, name='Alice'):
        super(LstmAgent, self).__init__()
        self.model = model
        self.args = args
        self.name = name
        self.human = False
        self.domain = domain.get_domain(args.domain)

    def _encode(self, inpt, dictionary):
        """A helper function that encodes the passed in words using the dictionary.

        inpt: is a list of strings.
        dictionary: prebuild mapping, see Dictionary in data.py
        """
        encoded = torch.LongTensor(dictionary.w2i(inpt)).unsqueeze(1)
        if self.model.device_id is not None:
            encoded = encoded.cuda(self.model.device_id)
        return encoded

    def _decode(self, out, dictionary):
        """A helper function that decodes indeces into English words.

        out: variable that contains an encoded utterance.
        dictionary: prebuild mapping, see Dictionary in data.py
        """
        return dictionary.i2w(out.data.squeeze(1).cpu())

    def feed_context(self, context):
        # the hidden state of all the pronounced words
        self.lang_hs = []
        # all the pronounced words
        self.words = []
        self.context = context
        # encoded context
        self.ctx = self._encode(context, self.model.context_dict)
        # hidded state of context
        self.ctx_h = self.model.forward_context(Variable(self.ctx))
        # current hidden state of the language rnn
        self.lang_h = self.model.zero_hid(1)

    def read(self, inpt, prefix_token='THEM:'):
        # print(inpt)
        inpt = self._encode(inpt, self.model.word_dict)

        lang_hs, self.lang_h = self.model.read(Variable(inpt), self.lang_h, self.ctx_h, prefix_token=prefix_token)
        # append new hidded states to the current list of the hidden states
        self.lang_hs.append(lang_hs.squeeze(1))
        # first add the special 'THEM:' token
        self.words.append(self.model.word2var('THEM:'))
        # then read the utterance
        self.words.append(Variable(inpt))
        assert (torch.cat(self.words).size()[0] == torch.cat(self.lang_hs).size()[0])

    def write(self):
        # generate a new utterance
        _, outs, self.lang_h, lang_hs = self.model.write(self.lang_h, self.ctx_h,
            100, self.args.temperature)

        # append new hidded states to the current list of the hidden states
        self.lang_hs.append(lang_hs)
        # first add the special 'YOU:' token
        self.words.append(self.model.word2var('YOU:'))
        # then append the utterance
        self.words.append(outs)
        assert (torch.cat(self.words).size()[0] == torch.cat(self.lang_hs).size()[0])
        # decode into English words
        return self._decode(outs, self.model.word_dict)

    def _choose(self, lang_hs=None, words=None, sample=False):
        # get all the possible choices
        choices = self.domain.generate_choices(self.context)
        # concatenate the list of the hidden states into one tensor
        lang_hs = lang_hs if lang_hs is not None else torch.cat(self.lang_hs)
        # concatenate all the words into one tensor
        words = words if words is not None else torch.cat(self.words)
        # logits for each of the item
        logits = self.model.generate_choice_logits(words, lang_hs, self.ctx_h)

        # construct probability distribution over only the valid choices
        choices_logits = []
        for i in range(self.domain.selection_length()):
            idxs = [self.model.item_dict.get_idx(c[i]) for c in choices]
            idxs = Variable(torch.from_numpy(np.array(idxs)))
            idxs = self.model.to_device(idxs)
            choices_logits.append(torch.gather(logits[i], 0, idxs).unsqueeze(1))

        choice_logit = torch.sum(torch.cat(choices_logits, 1), 1, keepdim=False)
        # subtract the max to softmax more stable
        choice_logit = choice_logit.sub(choice_logit.max().data[0])
        prob = F.softmax(choice_logit)
        if sample:
            # sample a choice
            idx = prob.multinomial().detach()
            logprob = F.log_softmax(choice_logit).gather(0, idx)
        else:
            # take the most probably choice
            _, idx = prob.max(0, keepdim=True)
            logprob = None

        p_agree = prob[idx.data[0]]

        # Pick only your choice
        return choices[idx.data[0]][:self.domain.selection_length()], logprob, p_agree.data[0]

    def choose(self):
        choice, _, _ = self._choose()
        return choice
