

import random
import argparse
import codecs
import os
import sys
import numpy as np

# Sequence - represents a sequence of hidden states and corresponding
# output variables.

class Sequence:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)

# HMM model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities
        e.g. {'happy': {'silent': '0.2', 'meow': '0.3', 'purr': '0.5'},
              'grumpy': {'silent': '0.5', 'meow': '0.4', 'purr': '0.1'},
              'hungry': {'silent': '0.2', 'meow': '0.6', 'purr': '0.2'}}"""



        self.transitions = transitions
        self.emissions = emissions

    ## part 1 - you do this.
    #TODO: Create unit test for load
    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""
        emit_dict = {}
        trans_dict = {}
        val_dict = {}
        emit = open((basename + '.emit')).readlines()
        for e in emit :
            #print(e)
            key,values,prob = e.split(' ')
            if key not in emit_dict:
                emit_dict[key] = {}
            emit_dict[key][values] = prob.strip()

        trans = open((basename + '.trans')).readlines()
        for t in trans :
            key,values,prob = t.split(' ')
            if key not in trans_dict:
                trans_dict[key] = {}
            trans_dict[key][values] = prob.strip()
        self.transitions = trans_dict
        self.emissions = emit_dict

   ## you do this.
    def generate(self, n):
        emit_seq = []
        trans_seq = []

        emit_seq_list = list(self.emissions.values())
        trans_seq_list = list(self.transitions.values())

        # ref: - https://sparkbyexamples.com/python/get-python-dictionary-values-to-list/
        emits_list = [e for e in emit_seq_list[1]]
        trans_list = [t for t in trans_seq_list[1]]

        emit_seq_list = [list(p.values())for p in emit_seq_list]
        trans_seq_list = [list(q.values())for q in trans_seq_list]
        """return an n-length Sequence by randomly sampling from this HMM."""
       ## starting state
        emit_state = np.random.choice(list(emits_list), p=emit_seq_list[0])
        trans_state = np.random.choice(list(trans_list), p=trans_seq_list[0])

        emit_seq.append(emit_state)
        trans_seq.append(trans_state)
        for i in range(n - 1):
            p_n = random.randint(0, len(trans_list) - 1)
            n_ts = np.random.choice(trans_list, p=trans_seq_list[p_n])
            n_es = np.random.choice(emits_list, p=emit_seq_list[p_n])

            emit_seq.append(n_es)
            trans_seq.append(n_ts)

        emit_seq = [str(t) for t in emit_seq]
        trans_seq = [str(s) for s in trans_seq]

        return Sequence(trans_seq, emit_seq)


    def forward(self, sequence):
        pass
    ## you do this: Implement the Viterbi algorithm. Given a Sequence with a list of emissions,
    ## determine the most likely sequence of states.






    def viterbi(self, sequence):
        pass
    ## You do this. Given a sequence with a list of emissions, fill in the most likely
    ## hidden states using the Viterbi algorithm.




if __name__ == '__main__':
    h = HMM()
    h.load('cat')
    # print(h.transitions)
    # print(h.emissions)

    # if len(sys.argv) < 3:
    #     print("Please provide all arguments")

    # basename = sys.argv[1]
    # func = sys.argv[2]
    # param = sys.argv[3]

    print(h.generate(20))


