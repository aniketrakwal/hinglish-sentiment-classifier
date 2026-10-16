"""Data preparation for the BiLSTM: tokenize, build a vocabulary, turn tweets into number tensors.

Flow for one tweet:
    raw text --clean_text()--> cleaned text --split(" ")--> tokens --vocab--> ids --pad/cut--> 40 ids
"""

from collections import Counter

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from preprocess import clean_text

PAD_ID = 0          # id of the padding token (filler that makes all tweets the same length)
UNK_ID = 1          # id of the "unknown word" token (any word not in the vocabulary)
MAX_LEN = 40        # tweets are cut / padded to this many tokens (EDA: covers 99.9% after cleaning)
MIN_FREQ = 2        # a word must appear at least twice in TRAIN to get its own id
BATCH_SIZE = 64

# The model needs numbers, so each label gets an id
LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}


def tokenize(text):
    """Clean a tweet with the shared clean_text(), then split it into tokens at single spaces."""
    cleaned = clean_text(text)
    if cleaned == "":           # "".split(" ") would give [""], a fake token; return no tokens instead
        return []
    return cleaned.split(" ")


def build_vocab(token_lists, min_freq=MIN_FREQ):
    """Map each frequent word to an id. Built from TRAIN tweets only (val/test must not leak in)."""
    counts = Counter()
    for tokens in token_lists:
        counts.update(tokens)                       # count how often each word appears
    word2idx = {"<pad>": PAD_ID, "<unk>": UNK_ID}   # ids 0 and 1 are reserved
    # most frequent first; ties broken alphabetically so the vocabulary is the same on every run
    for word, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if count >= min_freq:
            word2idx[word] = len(word2idx)          # next free id
    return word2idx


def encode(tokens, word2idx, max_len=MAX_LEN):
    """Turn a token list into (list of exactly max_len ids, real length)."""
    ids = [word2idx.get(token, UNK_ID) for token in tokens]   # unknown words become UNK_ID (1)
    ids = ids[:max_len]                                       # cut tweets that are too long
    if len(ids) == 0:
        ids = [UNK_ID]            # an empty tweet would have length 0, which the LSTM cannot handle
    length = len(ids)             # real length, before padding
    ids = ids + [PAD_ID] * (max_len - length)                 # fill the rest with 0s
    return ids, length


class TweetDataset(Dataset):
    """Holds all encoded tweets as tensors; PyTorch's DataLoader asks it for one example at a time."""

    def __init__(self, token_lists, labels, word2idx, max_len=MAX_LEN):
        encoded = [encode(tokens, word2idx, max_len) for tokens in token_lists]
        self.ids = torch.tensor([ids for ids, _ in encoded], dtype=torch.long)         # [N, max_len]
        self.lengths = torch.tensor([length for _, length in encoded], dtype=torch.long)  # [N]
        self.labels = torch.tensor([LABEL2ID[label] for label in labels], dtype=torch.long)  # [N]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.ids[index], self.lengths[index], self.labels[index]


def make_loader(dataset, shuffle, seed=42):
    """Batches of 64. Only the training loader shuffles; the seed makes the shuffle order repeatable."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, generator=generator)


def load_split(path):
    """Read one CSV and add a 'tokens' column (and the cleaned text, used later in the error file)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["cleaned_text"] = df["text"].apply(clean_text)
    df["tokens"] = df["text"].apply(tokenize)
    return df


def unk_share(token_lists, word2idx):
    """Share of tokens that are not in the vocabulary."""
    total = sum(len(tokens) for tokens in token_lists)
    unknown = sum(1 for tokens in token_lists for token in tokens if token not in word2idx)
    return unknown / total


def prepare_train_val(data_dir="data/processed"):
    """Load train and val, build the vocabulary from train, return everything the training script needs.
    (The test set is deliberately NOT loaded here.)"""
    train = load_split(f"{data_dir}/train.csv")
    val = load_split(f"{data_dir}/val.csv")
    word2idx = build_vocab(train["tokens"])
    print(f"vocabulary size: {len(word2idx)} (incl. <pad>, <unk>)")
    print(f"share of val tokens that are <unk>: {100 * unk_share(val['tokens'], word2idx):.1f}%")
    return {
        "train_df": train, "val_df": val, "word2idx": word2idx,
        "train_ds": TweetDataset(train["tokens"], train["label"], word2idx),
        "val_ds": TweetDataset(val["tokens"], val["label"], word2idx),
    }


def load_test_dataset(word2idx, data_dir="data/processed"):
    """Called only at the very end, once the final configuration has been chosen on validation."""
    test = load_split(f"{data_dir}/test.csv")
    return test, TweetDataset(test["tokens"], test["label"], word2idx)
