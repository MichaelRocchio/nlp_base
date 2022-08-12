import pandas as pd 
import numpy as np
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from nltk.corpus import words
from sklearn. feature_extraction.text import CountVectorizer 
from sklearn.metrics.pairwise import cosine_similarity 
from nltk.tokenize.api import TokenizerI
from nltk.util import ngrams 
import re
from string import punctuation 
import warnings
import nltk
from nltk.tokenize.api import TokenizerI
from nltk.util import ngrams
from abc import ABC, abstractmethod
from typing import Iterator, List, Tuple

from nltk.internals import overridden
from nltk.tokenize.util import string_span_tokenize


class TokenizerI(ABC):
    """
    A processing interface for tokenizing a string.
    Subclasses must define ``tokenize()`` or ``tokenize_sents()`` (or both).
    """

    @abstractmethod
    def tokenize(self, s: str) -> List[str]:
        """
        Return a tokenized copy of *s*.

        :rtype: List[str]
        """
        if overridden(self.tokenize_sents):
            return self.tokenize_sents([s])[0]


    def span_tokenize(self, s: str) -> Iterator[Tuple[int, int]]:
        """
        Identify the tokens using integer offsets ``(start_i, end_i)``,
        where ``s[start_i:end_i]`` is the corresponding token.

        :rtype: Iterator[Tuple[int, int]]
        """
        raise NotImplementedError()


    def tokenize_sents(self, strings: List[str]) -> List[List[str]]:
        """
        Apply ``self.tokenize()`` to each element of ``strings``.  I.e.:

            return [self.tokenize(s) for s in strings]

        :rtype: List[List[str]]
        """
        return [self.tokenize(s) for s in strings]


    def span_tokenize_sents(
        self, strings: List[str]
    ) -> Iterator[List[Tuple[int, int]]]:
        """
        Apply ``self.span_tokenize()`` to each element of ``strings``.  I.e.:

            return [self.span_tokenize(s) for s in strings]

        :yield: List[Tuple[int, int]]
        """
        for s in strings:
            yield list(self.span_tokenize(s))



class StringTokenizer(TokenizerI):
    """A tokenizer that divides a string into substrings by splitting
    on the specified string (defined in subclasses).
    """

    @property
    @abstractmethod
    def _string(self):
        raise NotImplementedError

    def tokenize(self, s):
        return s.split(self._string)


    def span_tokenize(self, s):
        yield from string_span_tokenize(s, self._string)


class SyllableTokenizer(TokenizerI):
    def __init__(self, lang="en", sonority_hierarchy=False):
        if not sonority_hierarchy and lang == "en":
            sonority_hierarchy = [
                "aeiouy",
                "lmnrw",
                "zvsf",
                "bcdgtkpqxhj",
            ]
        self.vowels = sonority_hierarchy[0]
        self.phoneme_map = {}
        for i, level in enumerate(sonority_hierarchy):
            for c in level:
                sonority_level = len(sonority_hierarchy) - i
                self.phoneme_map[c] = sonority_level
                self.phoneme_map[c.upper()] = sonority_level

    def assign_values(self, token):
        syllables_values = []
        for c in token:
            try:
                syllables_values.append((c, self.phoneme_map[c]))
            except KeyError:
                if c not in punctuation:
                    warnings.warn(
                        "Character not defined in sonority_hierarchy,"
                        " assigning as vowel: '{}'".format(c)
                    )
                    syllables_values.append((c, max(self.phoneme_map.values())))
                    self.vowels += c
                else:  # If it's a punctuation, assign -1.
                    syllables_values.append((c, -1))
        return syllables_values

    def validate_syllables(self, syllable_list):
        valid_syllables = []
        front = ""
        for i, syllable in enumerate(syllable_list):
            if syllable in punctuation:
                valid_syllables.append(syllable)
                continue
            if not re.search("|".join(self.vowels), syllable):
                if len(valid_syllables) == 0:
                    front += syllable
                else:
                    valid_syllables = valid_syllables[:-1] + [
                        valid_syllables[-1] + syllable
                    ]
            else:
                if len(valid_syllables) == 0:
                    valid_syllables.append(front + syllable)
                else:
                    valid_syllables.append(syllable)
        return valid_syllables

    def tokenize(self, token):
        syllables_values = self.assign_values(token)
        if sum(token.count(x) for x in self.vowels) <= 1:
            return [token]
        syllable_list = []
        syllable = syllables_values[0][0]
        for trigram in ngrams(syllables_values, n=3):
            phonemes, values = zip(*trigram)
            prev_value, focal_value, next_value = values
            focal_phoneme = phonemes[1]
            if focal_value == -1:
                syllable_list.append(syllable)
                syllable_list.append(focal_phoneme)
                syllable = ""
            elif prev_value >= focal_value == next_value:
                syllable += focal_phoneme
                syllable_list.append(syllable)
                syllable = ""
            elif prev_value > focal_value < next_value:
                syllable_list.append(syllable)
                syllable = ""
                syllable += focal_phoneme
            else:
                syllable += focal_phoneme
        syllable += syllables_values[-1][0]  # append last phoneme
        syllable_list.append(syllable)
        return self.validate_syllables(syllable_list)
    def get_cosine_similarity (feature_vec_1, feature_vec_2):
        return cosine_similarity (feature_vec_1, feature_vec_2)

    def token_list(list0):
        results=[] 
        last_val=[]
        last_token=''
        for i in list0:
            if i == '':
                results.append['']
            else:
                if i==last_val:
                    x.append(last_token)
                else:
                    last_val=i
                    i=''.join(e for e in i if e.isalnum()) 
                    last_token=tokenizer.tokenize(i)
                    results.append(last_token)
                    x.append(last_token)
        return results
