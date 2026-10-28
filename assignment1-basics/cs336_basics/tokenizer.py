"""
tokenizer for cs336 assignment1 test and for further reuse
junyu, 2026.10.26
"""

from collections.abc import Iterable, Iterator
import os
import pickle
import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from typing import IO, Any, BinaryIO

class Tokenizer():
    """BPE tokenizer class"""
    def __init__(self, vocab, merges, special_tokens=None): 
        """
            Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.
            This function should accept the following parameters:
            vocab: dict[int, bytes]  
            merges: list[tuple[bytes, bytes]]  
            special_tokens: list[str] | None = None 
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.revert_vocab = {v:k for k, v in self.vocab.items()}


    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
            Class method
            that constructs and returns a Tokenizer from a serialized vocabulary and list of merges (in the
            same format that your BPE training code output) and (optionally) a list of special tokens.
            This method should accept the following additional parameters:
            vocab_filepath: str  
            merges_filepath: str  
            special_tokens: list[str] | None = None
        """
        vocab, merges = None, None
        with open(vocab_filepath, "r") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "r") as g:
            merges = pickle.load(g)

        return cls(vocab, merges, special_tokens)


    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        res = []

        text_bytes = list(text.encode('utf-8'))
        n_encode = len(text_bytes)
        ind = 0
        while ind < n_encode:
            if ind == n_encode - 1:
                res.append(self.revert_vocab[text_bytes[ind]])


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. 
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        """

        for text in iterable:
            return self.encode(text)


    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        res = ''.join([self.vocab[id].decode('utf-8') for id in ids])
        return res

