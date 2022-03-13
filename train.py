import os
import pickle
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import ArticlesDataModule
from model import LSTMModel, TransformerModel


class Trainer(object):
    def __init__(
            self,
            model: str,
            max_epochs: int,
            batch_size: int,
            lr: int,
            data_root: str,
            columns: List[str],
            input_size: int,
            hidden_size: int) -> None:

        self.max_epochs = max_epochs

        data_module = ArticlesDataModule(
            data_root=data_root,
            columns=columns,
            batch_size=batch_size)
        self.train_data = data_module.train_data()
        self.val_data = data_module.val_data()
        ntokens = data_module.ntokens
        seq_len = data_module.seq_len

        assert model in [
            "lstm", "transformer"], "Wrong model, choose from ['lstm', 'transformer']"
        self.model_name = model
        if model == "lstm":
            self.model = LSTMModel(
                input_size=input_size,
                hidden_size=hidden_size,
                seq_len=seq_len,
                num_embeddings=ntokens)
        elif model == "transformer":
            self.model = TransformerModel(
                input_size=input_size,
                hidden_size=hidden_size,
                num_embeddings=ntokens,
                nhead=2,
                nlayer=3,
                seq_len=seq_len)

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.writter = SummaryWriter()

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode='min',
            patience=1)

    def train_model(self) -> None:
        self.global_step = 0
        train_loss_d, train_acc_d, self.val_loss_d, self.val_acc_d = {}, {}, {}, {}
        for epoch in range(self.max_epochs):
            self.model.train()
            total_loss = 0
            total_acc = 0
            for batch in tqdm(self.train_data):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.optimizer.zero_grad()
                output = self.model(batch).squeeze()
                loss = self.criterion(output, batch["target"].float())
                loss.backward()
                self.optimizer.step()
                batch_size = batch["source"].shape[0]
                self.model.init_hidden(batch_size=batch_size)
                train_acc = self.binary_acc(output, batch["target"].float())
                total_loss += loss.item()
                total_acc += train_acc
                self.writter.add_scalar(
                    "train/train_accuracy", train_acc, self.global_step)
                self.writter.add_scalar(
                    "train/train_loss", loss.item(), self.global_step)
                self.global_step += 1
            self.evaluate_model(epoch=epoch)
            meaned_loss = total_loss / len(self.train_data)
            meaned_acc = total_acc / len(self.train_data)
            train_loss_d[epoch] = meaned_loss
            train_acc_d[epoch] = meaned_acc
            print(f'Epoch {epoch} Training loss: {meaned_loss}')
        self.save_result_dict(train_loss_d, "train_loss")
        self.save_result_dict(train_acc_d, "train_acc")
        self.save_result_dict(self.val_loss_d, "val_loss")
        self.save_result_dict(self.val_acc_d, "val_acc")
        self.writter.close()

    def evaluate_model(self, epoch: int) -> None:
        self.model.eval()
        total_loss = 0
        val_acc = 0
        with torch.no_grad():
            for batch in tqdm(self.val_data):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                output = self.model(batch).squeeze()
                loss = self.criterion(output, batch["target"].float())
                batch_size = batch["source"].shape[0]
                self.model.init_hidden(batch_size=batch_size)
                total_loss += loss.item()
                val_acc += self.binary_acc(output, batch["target"].float())
            meaned_loss = total_loss / len(self.val_data)
            meaned_acc = val_acc / len(self.val_data)
            self.writter.add_scalar(
                "val/val_loss", meaned_loss, self.global_step)
            self.writter.add_scalar("val/val_accuracy", meaned_acc, epoch)
            self.val_loss_d[epoch] = meaned_loss
            self.val_acc_d[epoch] = meaned_acc
            print(f'Epoch {epoch} Validation loss: {meaned_loss}')

    def binary_acc(self, model_output, target) -> float:
        out_tag = torch.round(torch.sigmoid(model_output))
        correct_results_sum = (out_tag == target).sum().float()
        acc = correct_results_sum / target.shape[0]
        acc = acc * 100
        return acc.item()

    def save_result_dict(self, results: Dict, res_name: str) -> None:
        results_folder = "results"
        if not os.path.isdir(results_folder):
            os.mkdir(results_folder)
        res_name = self.model_name + "_" + res_name + ".pkl"
        results_path = os.path.join(results_folder, res_name)
        with open(results_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Results saved to {results_path}")


if __name__ == '__main__':
    trainer = Trainer(
        model="transformer",
        max_epochs=100,
        batch_size=16,
        lr=0.001,
        data_root="data",
        columns=["lemma_title", "lemma_description"],  # lemma_maintext
        input_size=16,
        hidden_size=16)

    trainer.train_model()
