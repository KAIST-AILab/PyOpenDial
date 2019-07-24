from parlai.core.params import ParlaiParser
from example_domains.chitchat.agent import TransformerAgent


parser = ParlaiParser(add_model_args=True)
parser.set_defaults(model='example_domains.chitchat.agent:TransformerAgent',
                    sample=False,
                    wild_mode=False,
                    replace_repeat=False,
                    replace_ngram=False,
                    detokenize=False,
                    emoji_prob=0,
                    add_questions=0,
                    clean_emoji=False,
                    check_grammar=False,
                    correct_generative=False,
                    split_into_sentences=False,

                    max_seq_len=256,
                    beam_size=3,
                    annealing_topk=None,
                    annealing=0.6,
                    length_penalty=0.6)

opt = parser.parse_args()
agent = TransformerAgent(opt)


def observe_and_act(u_u):
    agent.observe({
        'text': u_u,
        'episode_done': False
    })

    return agent.act()['text']
