import unittest

from HMM import *

class MyTestCase(unittest.TestCase):
    def test_load(self):
        h = HMM()
        h.load('cat')
        trans_dict_test = {'#': {'happy': '0.5', 'grumpy': '0.5', 'hungry': '0'}, 'happy': {'happy': '0.5', 'grumpy': '0.1', 'hungry': '0.4'}, 'grumpy': {'happy': '0.6', 'grumpy': '0.3', 'hungry': '0.1'}, 'hungry': {'happy': '0.1', 'grumpy': '0.6', 'hungry': '0.3'}}

        emiss_dict_test = {'happy': {'silent': '0.2', 'meow': '0.3', 'purr': '0.5'}, 'grumpy': {'silent': '0.5', 'meow': '0.4', 'purr': '0.1'}, 'hungry': {'silent': '0.2', 'meow': '0.6', 'purr': '0.2'}}

        self.assertEqual(trans_dict_test, h.transitions)
        self.assertEqual(h.emissions, emiss_dict_test)



