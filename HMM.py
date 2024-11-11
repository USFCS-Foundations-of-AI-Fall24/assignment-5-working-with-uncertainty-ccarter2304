

import random
import argparse
import codecs
import os
import sys

import numpy

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
        """return an n-length Sequence by randomly sampling from this HMM."""
        pass

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
    print(h.transitions)
    print(h.emissions)



