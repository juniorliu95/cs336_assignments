"""
tokenizer for cs336 assignment1 test and for further reuse
junyu, 2026.10.26
"""

from collections.abc import Iterable, Iterator
import regex as re
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
        self.special_tokens = []
        # for death use case like special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"]
        if special_tokens:
            self.special_tokens = sorted(special_tokens, key=lambda x: -len(x))
        
        # # Add missing special tokens to the vocabulary
        # if self.special_tokens:
        #     next_id = max(self.vocab.keys()) + 1 if self.vocab else 256
        #     existing_bytes = set(self.vocab.values())
        #     import pdb;pdb.set_trace()
        #     for special_token in self.special_tokens:
        #         st_bytes = special_token.encode("utf-8")
        #         if st_bytes not in existing_bytes:
        #             self.vocab[next_id] = st_bytes
        #             existing_bytes.add(st_bytes)
        #             next_id += 1

        self.revert_vocab = {}
        for k, v in self.vocab.items():
            if v not in self.revert_vocab:
                self.revert_vocab[v] = k
            elif k < self.revert_vocab[v]: # handle duplicated values, keep smaller key
                self.revert_vocab[v] = k


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
        data = []
        if self.special_tokens:
            # there can be `|` in special tokens
            escaped_specials = [re.escape(tok) for tok in self.special_tokens]
            special_matches = re.finditer("|".join(escaped_specials), text)
            str_begin = 0
            for special_match in special_matches:
                special_token = special_match.group()
                special_byte =special_token.encode("utf-8")
                special_id = self.revert_vocab[special_byte]
                sp_start, sp_end = special_match.start(), special_match.end()
                if sp_start > str_begin:
                    subtext = text[str_begin:sp_start]
                    data.append(subtext)
                data.append(special_id)

                str_begin = sp_end
            if str_begin < len(text):
                subtext = text[str_begin:]
                data.append(subtext)
        else:
            data = [text]
        
        # pre-tokenization and init the counter
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        for subtext in data:
            if isinstance(subtext, int):
                res.append(subtext)
                continue

            matches = re.finditer(PAT, subtext)

            for match in matches:
                match_text = match.group()
                splited_string = [bytes([b]) for b in match_text.encode("utf-8")]
                splited_res = [self.revert_vocab[ss] for ss in splited_string]
                # make the logic similar to bpe training. each loop take the highest merge
                while len(splited_string) >= 2:
                    candidates = {} # merge_pair_ind: merge_rank
                    for i in range(len(splited_string) - 1):
                        merge_pair = (splited_string[i], splited_string[i+1])
                        if merge_pair in self.merges:
                            candidates[i] = self.merges.index(merge_pair)
                    if not candidates:
                        break
                    best_merge_pair_ind = min(candidates, key=lambda x: candidates[x])
                    splited_string[best_merge_pair_ind] += splited_string[best_merge_pair_ind+1]
                    splited_string.pop(best_merge_pair_ind+1)
                    splited_res[best_merge_pair_ind] = self.revert_vocab[splited_string[best_merge_pair_ind]]
                    splited_res.pop(best_merge_pair_ind+1)
                res.extend(splited_res)
        return res

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. 
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        """

        for text in iterable:
            yield from self.encode(text)


    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        token_bytes = b"".join(self.vocab[idx] for idx in ids)
        # import pdb;pdb.set_trace()
        return token_bytes.decode("utf-8", errors="replace") # add error control based on the handout

