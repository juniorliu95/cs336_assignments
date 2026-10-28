"""
bpe for cs336 assignment1 test and for further reuse
junyu, 2026.10.26
"""
import os
import regex as re
from typing import List, Any
from collections import Counter

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
        data = f.readlines()

    # pre-tokenization to remove dirty items
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    word_counter = Counter()

    for subtext in data: 
        matches = re.finditer(PAT, subtext)
        
        for match in matches:
            match_text = match.group()
            word_counter[match_text] += 1

    pre_counter = Counter()
    for item in list(word_counter):
        item_text, count = item
        for i in range(len(item_text)-1):
            pre_counter[(item_text[i], item_text[i+1])] += 1

    # training steps
    while pre_counter and len(vocab) < vocab_size:
        merge_step(word_counter, pre_counter, merges)

    
    return vocab, merges


def merge_step(word_counter: Counter, pre_counter: Counter, merges: List[Any]) -> None:
    max_value = max(pre_counter.values())
    max_item = max([item for item, count in pre_counter.items() if count == max_value])
    max_item_bytes = (max_item[0].encode("utf-8"), max_item[1].encode("utf-8"))
    merges.append(max_item_bytes)

    # remove the max item, find new pairs including the max item, add to the pre_counter
    pre_counter.pop(max_item)

    new_token = ''.join(max_item)
    for item in list(word_counter):
        item_text, count = item
        new_token_inds = [m.start() for m in re.finditer(new_token, item_text)]
        for new_token_ind in new_token_inds:
            

