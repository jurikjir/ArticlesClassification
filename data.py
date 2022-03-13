import ast
import itertools
import os
import pickle
from types import ModuleType
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchtyping import TensorType
from tqdm import tqdm


class ArticlesDataModule(object):
    """
    Data module which contains dataloaders for dataset.
    Module checks if vocabulary and translated dataset are present
    in preprocessed folder. If they are not, this module creates them.
    """

    def __init__(self, data_root: str, columns: List[str], batch_size: int) -> None:

        processed_root = os.path.join(data_root, "processed")
        if not os.path.isdir(processed_root):
            os.mkdir(processed_root)

        prefix = "".join([name.split("_")[1][0] for name in columns])
        dataset_filename = prefix + "_dataset.pt"
        vocab_filename = prefix + "_vocab.pkl"
        dataset_path = os.path.join(processed_root, dataset_filename)
        vocab_path = os.path.join(processed_root, vocab_filename)
        exists_dataset = os.path.exists(dataset_path)
        exists_vocab = os.path.exists(vocab_path)
        if not (exists_dataset & exists_vocab):
            encoder = TextEncoder(data_root=data_root, columns=columns)
            encoder.build_vocab(vocab_path=vocab_path)
            encoder.convert_text(dataset_path=dataset_path)

        train_dataset = ArticlesDataset(dataset_path, split="train")
        val_dataset = ArticlesDataset(dataset_path, split="val")

        self.ntokens = train_dataset.ntokens
        self.seq_len = train_dataset.seq_len

        self.train_dataloader = ArticlesDataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )

        self.val_dataloader = ArticlesDataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True
        )

    def train_data(self) -> ModuleType:
        return self.train_dataloader

    def val_data(self) -> ModuleType:
        return self.val_dataloader


class ArticlesDataLoader(DataLoader):
    """
    Inherit from torch DataLoader which gathers datasamples from dataset
    to batches. Next passes through ntokens and seq_len variables which are
    needed in training module. 
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def ntokens(self):
        return self.dataset.ntokens

    @property
    def seq_len(self):
        return self.dataset.seq_len


class ArticlesDataset(Dataset):
    """
    Dataset module for Articles, it takes care of dataset loading,
    splitting and data sampling.
    """
    def __init__(self, dataset_path: str, split: str) -> None:
        super().__init__()
        target_path = dataset_path.split("_")[0] + "_target.pt"
        vocab_path = dataset_path.split("_")[0] + "_vocab.pkl"
        dataset = torch.load(dataset_path)
        target = torch.load(target_path)
        dataset_len = len(dataset)
        train_len = round(dataset_len * 0.85)
        if split == "train":
            self.dataset = dataset[:train_len]
            self.target = target[:train_len]
        elif split == "val":
            self.dataset = dataset[train_len:]
            self.target = target[train_len:]
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)

    def __getitem__(self, idx: int) -> Dict[str, TensorType]:
        item = {"source": self.dataset[idx], "target": self.target[idx]}
        return item

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def ntokens(self) -> int:
        """
        Needed for setting dimension of embedding tensor later in model
        """
        return len(self.vocab)

    @property
    def seq_len(self):
        """
        Needed for setting dimension of decoder later in model
        """
        return self.dataset.shape[1]


class TextEncoder(object):
    def __init__(self, data_root: str, columns):
        """
        This module is responsible for vocabulary
        creation and dataset translation
        """
        self.data_root = data_root
        self.data_file = os.path.join(data_root, "Relevant vs Irrelevant.xlsx")
        self.processed_root = os.path.join(data_root, "processed")
        self.columns = columns
        self.data = pd.read_excel(self.data_file, engine="openpyxl")
        self.lang_tokens = list(self.data["language"].unique())
        # special tokens which determines start of specific subpart of sequence
        # for example maintext is bounded by: 
        # <ms> and <me> which means <maintextstart> and <maintextend>
        self.special_tokens = {
            "language": ["<ls>", "<le>"],
            "title": ["<ts>", "<te>"],
            "maintext": ["<ms>", "<me>"],
            "description": ["<ds>", "<de>"],
            "padd": ["<pad>"]
        }

    def convert_text(self, dataset_path: str) -> None:
        """
        Convert words string representation to numerical tokens 
        with vocabulary mapping and save in torch tensor format.
        """
        print("Converting dataset...")
        seqs = []
        self.all_columns = self.columns + ["language"]
        for row in tqdm(range(len(self.data))):
            seq = []
            for column in self.all_columns:
                column_name = column.split("_")[1] if len(
                    column.split("_")) == 2 else column
                seq.append(self.vocab[self.special_tokens[column_name][0]])
                if column == "language":
                    content = [self.data.iloc[row][column]]
                else:
                    content = ast.literal_eval(self.data.iloc[row][column])
                seq.extend([self.vocab[token] for token in content])
                seq.append(self.vocab[self.special_tokens[column_name][1]])
            seqs.append(seq)
        max_seq = max([len(seq) for seq in seqs])
        pad_lens = [max_seq - len(seq) for seq in seqs]
        padd_seqs = [seq + [self.vocab["<pad>"]] * pad_lens[i]
                     for i, seq in enumerate(seqs)]
        source = torch.tensor(padd_seqs)
        torch.save(obj=source, f=dataset_path)
        target = torch.tensor(list(self.data["target"]))
        target_path = dataset_path.split("_")[0] + "_target.pt"
        torch.save(obj=target, f=target_path)
        print(f"Dataset successfully created and saved to {dataset_path}")

    def build_vocab(self, vocab_path: str) -> None:
        """
        Gather all unique words in datset and create vocabulary,
        each unique word is mapped to unique number. Save vocabulary.
        """
        column_values = []
        err_cnt = 0
        self.maxlens = {column: 0 for column in self.columns}
        print("Building vocab...")
        for column in tqdm(self.columns):
            for i, lemma in enumerate(self.data[column]):
                # approx 20 of the values are not processed correctly -> EOL while
                # converting from str to list, dataset contains 3160 samples,
                # so these 20 will be dropped
                try:
                    lemma_list = ast.literal_eval(lemma)
                    column_values.extend(lemma_list)
                    # get len of the longest sequence in given column
                    # this will be used later in tokenize_text for padding
                    if len(lemma_list) > self.maxlens[column]:
                        self.maxlens[column] = len(lemma_list)
                except:
                    self.data.drop(
                        index=self.data[self.data[column] == lemma].index, inplace=True)
                    err_cnt += 1
                    continue
        special_tokens = list(itertools.chain(
            *list(self.special_tokens.values())))
        column_values.extend(self.lang_tokens + special_tokens)
        unique_chars = list(set(column_values))
        self.vocab = {token: idx for idx, token in enumerate(unique_chars)}
        with open(vocab_path, "wb") as f:
            pickle.dump(self.vocab, f)
        print(f"Vocab built. Number of errors: {err_cnt}")
