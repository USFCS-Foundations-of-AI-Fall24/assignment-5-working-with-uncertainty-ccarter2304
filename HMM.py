

import random
import argparse
import codecs
import os
import sys
from turtledemo.penrose import start

import numpy as np
from jinja2.sandbox import MAX_RANGE


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
    ## global variable for lander cases
    LANDER = False
    #TODO: Create unit test for load
    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""
        if basename == 'lander' :
            HMM.LANDER = True
        emit_dict = {}
        trans_dict = {}
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
   #One cool thing we can do with an HMM is Monte Carlo simulation.
    # We'll do this using generate. So implement that next.
    # It should take an integer n, and return a Sequence of length n.
    # To generate this, start in the initial state and
    # repeatedly select successor states at random, using the transition probability as a weight,
    # and then select an emission, using the emission probability as a weight.
    # You may find either numpy.random.choice or random.choices very helpful here. B
    # e sure that you are using the transition probabilities to determine the next state, and not a uniform distribution!
    def generate(self, n):
        emis_states = []
        trans_states = []
        # get the start state from "#"
        start_states = [v for v in self.transitions.get("#")]
        start_weights = []
        for state in start_states :
            start_weights.append(self.transitions.get("#", {}).get(state))
        cur_state = np.random.choice(start_states, p=start_weights)
        for i in range(n) :
            ## emissions
            next_emis_states = [v for v in self.emissions.get(cur_state)]
            next_emis_weights = []
            for state in next_emis_states:
                next_emis_weights.append(self.emissions.get(cur_state, {}).get(state))
            ## transitions
            next_states = [v for v in self.transitions.get(cur_state)]
            next_weights = []
            for state in next_states:
                next_weights.append(self.transitions.get(cur_state, {}).get(state))

            emis_states.append(np.random.choice(next_emis_states, p=next_emis_weights))
            cur_state = np.random.choice(next_states, p=next_weights)
            trans_states.append(cur_state)
        return Sequence(trans_states, emis_states)

    ## determine the most likely sequence of states.
    def forward(self, sequence):
        # setup matrix

        m = []
        intial = [v for v in self.transitions]
        seq_row = [""] * 2 + [str(s) for s in sequence.outputseq]
        m.append(seq_row)
        for i in intial :
            if i == '#':
                i_row = [i] + [1.0] + [0] * len(sequence.outputseq)
            else :
                i_row = [i] + [0] * (len(sequence.outputseq) + 1)
            m.append(i_row)
        # compute values for day 1
        for k in range(len(intial) - 1) :
            if self.emissions.get(m[k + 2][0], {}).get(m[0][2]) is not None:
                m[k + 2][2] = float(self.emissions.get(m[k + 2][0], {}).get(m[0][2])) * float(self.transitions.get("#", {}).get(m[k + 2][0]))
            else :
                m[k + 2][2] = 0
        # compute values for day 2,...n
        intial.remove("#")
        for j in range(3, len(sequence) + 2) :
            state_index = 2
            for state in intial :
                s_sum = 0
                s2_indx = 2
                for s2 in intial :
                    # ex: P(silent | happy) * P(happy | happy) * P(happy)
                    # P(silent | happy) = value from emissions
                    # P(happy | happy) = value from transitions
                    # P(happy) = value at matrix at previous column

                    ##Check for none type due to not emissions and values not aligning in all cases
                    if self.emissions.get(state, {}).get(m[0][j]) is not None :

                        s_sum += (float(self.emissions.get(state, {}).get(m[0][j])) *
                                     float(self.transitions.get(s2, {}).get(state)) *
                                     float(m[s2_indx][j - 1]))
                    else :
                        s_sum += 0.0 * float(self.transitions.get(s2, {}).get(state)) * float(m[s2_indx][j - 1])
                    s2_indx += 1
                m[state_index][j] = s_sum
                state_index += 1
        ## determine the most probable state
        #get the value at the last column in each row
        probable_state_val = m[2][len(sequence.outputseq) + 1]
        probable_state_indx = 2
        for i in range (3, len(sequence.outputseq) + 1) :
            if m[i][len(sequence.outputseq) + 1] > probable_state_val :
                probable_state_val = m[i][len(sequence.outputseq) + 1]
                probable_state_indx = i
        probable_state = m[probable_state_indx][0]

        print("The most probable state: ",  probable_state, " with probability: ",  probable_state_val)
        ## for lander
        landable_states = ['4,3', '4,4', '3,4', '2,5', '5,5']
        if HMM.LANDER :
            if probable_state in landable_states :
                print("Safe to land")
            else :
                print("Not to safe land")
    ## you do this: Implement the Viterbi algorithm. Given a Sequence with a list of emissions,

    def viterbi(self, sequence):
        ## create setup matrix
        m = []
        ## backpointer matrix
        b = []

        intial = [v for v in self.transitions]
        seq_row = [""] * 2 + [str(s) for s in sequence.outputseq]
        m.append(seq_row)
        b.append(seq_row)
        for i in intial :
            if i == '#':
                i_row = [i] + [1.0] + [0] * len(sequence.outputseq)
            else :
                i_row = [i] + [0] * (len(sequence.outputseq) + 1)
            m.append(i_row)
            b.append([i] + [0] * (len(sequence.outputseq) + 1))
        # compute values for day 1
        for k in range(len(intial) - 2) :
            if self.emissions.get(m[k + 2][0], {}).get(m[0][2]) is not None:
                m[k + 2][2] = float(self.emissions.get(m[k + 2][0], {}).get(m[0][2])) * float(self.transitions.get("#", {}).get(m[k + 2][0]))
            else :
                m[k + 2][2] = 0
        # compute values for day 2..
        intial.remove("#")
        for j in range(3, len(sequence) + 2) :
            state_index = 2
            for state in intial :
                max_val = 0
                s2_indx = 2
                max_indx = 0
                for s2 in intial :
                    # find the value and set the max
                    if self.emissions.get(state, {}).get(m[0][j]) is not None:
                        val = (float(self.emissions.get(state, {}).get(m[0][j])) *
                                     float(self.transitions.get(s2, {}).get(state)) *
                                     float(m[s2_indx][j - 1]))
                    else :
                        val = 0.0  * float(self.transitions.get(s2, {}).get(state)) * float(m[s2_indx][j - 1])

                    if val > max_val :
                        max_val = val
                        max_indx = s2_indx
                    s2_indx += 1
                m[state_index][j] = max_val
                b[state_index][j] = max_indx - 1
                state_index += 1

        # #get the most likely state
        probable_state_val = m[2][len(sequence.outputseq) + 1]
        probable_state_indx = 2
        for i in range (3, len(sequence.outputseq) + 1) :
            if m[i][len(sequence.outputseq) + 1] > probable_state_val :
                probable_state_val = m[i][len(sequence.outputseq) + 1]
                probable_state_indx = i
        probable_state = m[probable_state_indx][0]
        ## traverse back using the backpointers
        state_list = [probable_state]

        current_index = probable_state_indx
        for j in range(len(sequence.outputseq) + 1, 2, -1):
            current_index = b[current_index][j] + 1
            state_list.append(m[current_index][0])
        #reverse the list
        state_list.reverse()
        print(state_list)


    ## You do this. Given a sequence with a list of emissions, fill in the most likely
    ## hidden states using the Viterbi algorithm.




if __name__ == '__main__':
    #TODO: Implement command line arguments
    #TODO: Finish lander emit and trans files
    h = HMM()
    h.load('partofspeech')
    # print(h.transitions)
    # print(h.emissions)

    # if len(sys.argv) < 3:
    #     print("Please provide all arguments")

    # basename = sys.argv[1]
    # func = sys.argv[2]
    # param = sys.argv[3]

    h.viterbi(h.generate(3))


