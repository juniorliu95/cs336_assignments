"""
bpe for cs336 assignment1 test and for further reuse
junyu, 2026.10.26
"""
import os
import regex as re
from typing import List, Dict, Any
from collections import Counter

# from .pretokenization import find_chunk_boundaries

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # add initial tokens
    vocab = {i: bytes([i]) for i in range(256)}

    # add special tokens
    next_id = 256
    for token_str in special_tokens:
        token_bytes = token_str.encode("utf-8")
        vocab[next_id] = token_bytes
        next_id += 1

    merges = []

    with open(input_path, "r") as f:
        # data = f.readlines() # ! readlines will ignore the \n!
        data = f.read()

    if special_tokens:
        data = re.split("|".join(special_tokens), data)

    # pre-tokenization and init the counter
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    splited_data = []
    pre_counter = Counter()

    for subtext in data:
        matches = re.finditer(PAT, subtext)
        
        for match in matches:
            match_text = match.group()
            splited_string = [bytes([b]) for b in match_text.encode("utf-8")]
            splited_data.append(splited_string)

            for i in range(len(splited_string)-1): 
                pre_counter[(splited_string[i], splited_string[i+1])] += 1

    # training steps
    while pre_counter and splited_data and len(vocab) < vocab_size:
        merge_step(splited_data, pre_counter, vocab, merges)

    return vocab, merges


def merge_step(splited_data: List[Any], pre_counter: Counter, vocab: Dict, merges: List[Any]) -> None:
    # find out current max
    max_value = max(pre_counter.values())
    max_items = [k for k, v in pre_counter.items() if v==max_value]
    merge_item = max(max_items)
    new_token = merge_item[0] + merge_item[1]
    # import pdb;pdb.set_trace()
    merges.append(merge_item)
    vocab[len(vocab)] = new_token

    # update pre_counter and splited_data
    pre_counter.pop(merge_item)
    i = 0
    splited_to_remove = []
    while i < len(splited_data):
        splited_d = splited_data[i]
        if merge_item[0] in splited_d and merge_item[1] in splited_d:
            j = 0
            while j <len(splited_d) - 1:
                if (splited_d[j], splited_d[j+1]) == merge_item:
                    # left side update
                    if j-1 >= 0:
                        left_pair = (splited_d[j-1], splited_d[j])
                        pre_counter[left_pair] -= 1
                        pre_counter[(splited_d[j-1], new_token)] += 1 # count + 1 before the actual text change for convinience
                        if pre_counter[left_pair] == 0:
                            del pre_counter[left_pair]
                    # right side update
                    if j+2 < len(splited_d):
                        right_pair = (splited_d[j+1], splited_d[j+2])
                        pre_counter[right_pair] -= 1
                        pre_counter[(new_token, splited_d[j+2])] += 1 # count + 1 before the actual text change for convinience                            
                        if pre_counter[right_pair] == 0:
                            del pre_counter[right_pair] 
                    splited_d[j+1] = new_token
                    del splited_d[j]
                else:
                    j += 1
        # we dont need that item anymore if no new token can be gen
        if len(splited_d) == 1:
            del splited_data[i]
        else:
            i += 1
            

