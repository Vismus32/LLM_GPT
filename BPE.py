import sys
import torch
import numpy
from collections import Counter

class BPE():
    def __init__(self,vocab_size:int):
        self.vocab_size = vocab_size
    def fit(self, text:str):
        uniq_tokens = []
        tokens = []

        for i in range(len(text)):
            tokens.append(text[i])
            if text[i] not in uniq_tokens:
                uniq_tokens.append(text[i])
        uniq_tokens = sorted(uniq_tokens)

        while len(uniq_tokens) != self.vocab_size:
            pairs = {}
            for i in range(len(tokens)-1):
                pair = tokens[i:i+2]
                if pair in pairs:
                    pairs[pair]+=1
                else:
                    pairs[pair] = 1
            max_pair = None
            max_count = 0
            for pair in pairs:
                if pairs[pair] > max_count:
                    max_count = pairs[pair]
                    max_pair = pair
            

            new_tokens = []

            i = 0

            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append(max_pair)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens
            #TODO напиши коменты, всё стало не понятно



