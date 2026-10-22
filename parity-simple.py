import numpy as np
import torch
import torch.nn as nn
import lightning.pytorch as pl
import torchmetrics
import torchvision
from torchinfo import summary
from torchview import draw_graph
from IPython.display import display
import sympy as sp
sp.init_printing(use_latex=True)
import contextlib
import pandas as pd
import matplotlib.pyplot as plt
import logging
import wandb
import json
import warnings
from pathlib import Path
import argparse
import os
import urllib
import random
from torch.nn.utils.rnn import pad_sequence

HERE = Path(__file__).resolve().parent
EXPERIMENT = Path(__file__).name

USE_RNN = True
USE_CUSTOM = False
PARITY_PROBLEM = True
SILENT = False

PROJECT = "OL4-Parity" if PARITY_PROBLEM else "OL4-Translation"

output_context = (
    contextlib.redirect_stdout(open(os.devnull, "w"))
    if SILENT
    else contextlib.nullcontext()
)

error_context = (
    contextlib.redirect_stderr(open(os.devnull, "w"))
    if SILENT
    else contextlib.nullcontext()
)

torch.set_float32_matmul_precision('medium')


class CharacterTranslationDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size=100,
                 n_sequences=1000,
                 val_split=0.2,
                 num_workers=4,
                 data_url="https://raw.githubusercontent.com/luisroque/deep-learning-articles/main/data/eng-por.txt",
                 **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.n_sequences = n_sequences
        self.n_timesteps = None
        self.val_split = val_split
        self.num_workers = num_workers
        self.data_url = data_url
        self.input_shape = None
        self.output_shape = None
        self.data_train = None
        self.data_val = None
        self.data_test = None

    def encode_seq(self, x, mapping, max_length=0):
        # String to integer
        return [mapping['<START>']] + \
            [mapping[i] for i in list(x)] + \
            [mapping['<STOP>']] + \
            [0] * (max_length - len(list(x)) - 2)

    def decode_seq(self, x, mapping):
        # Integer-to-string
        try:
            idx = list(x).index(2)  # Stop token?
        except:
            idx = len(list(x))  # No stop token found
        return ''.join([mapping[i] for i in list(x)[0:idx]])

    def setup(self, stage: str):
        if (stage == 'fit' or \
            stage == 'validate' or
            stage == 'test' or \
            stage == 'predict') and \
                not (self.data_train and self.data_val and self.data_test):
            data = []
            with urllib.request.urlopen(self.data_url) as raw_data:
                for line in raw_data:
                    data.append(line.decode("utf-8").split('\t')[0:2])
            data = np.array(data)
            # Data is generally sorted by -length- (smaller sentences first)
            if self.n_sequences is not None:
                data = data[0:self.n_sequences]
            self.n_timesteps = np.max([len(i) for i in data.flatten()]) + 2  # Add start/stop
            # Produce mappings with -unshuffled- data for consistency
            self.i_to_c_src = ['', '<START>', '<STOP>'] + list({char for word in data[:, 0] for char in word})
            self.c_to_i_src = {self.i_to_c_src[i]: i for i in range(len(self.i_to_c_src))}
            self.i_to_c_src[1] = self.i_to_c_src[2] = ''
            self.i_to_c_dst = ['', '<START>', '<STOP>'] + list({char for word in data[:, 1] for char in word})
            self.c_to_i_dst = {self.i_to_c_dst[i]: i for i in range(len(self.i_to_c_dst))}
            self.i_to_c_dst[1] = self.i_to_c_dst[2] = ''
            # Shuffle data for training/validation splits (monte carlo sampling)
            rng = np.random.default_rng()
            rng.shuffle(data)
            X = np.vstack([self.encode_seq(x, self.c_to_i_src, self.n_timesteps) for x in data[:, 0]])
            Y = np.vstack([self.encode_seq(x, self.c_to_i_dst, self.n_timesteps) for x in data[:, 1]])
            self.input_shape = X.shape[1:]
            self.output_shape = Y.shape[1:]
            split_point = int(X.shape[0] * (1.0 - self.val_split))
            enc_x_train = X[:split_point]
            enc_x_val = X[split_point:]
            dec_x_train = Y[:, 0:-1][:split_point]
            dec_x_val = Y[:, 0:-1][split_point:]
            dec_y_train = Y[:, 1:][:split_point]
            dec_y_val = Y[:, 1:][split_point:]
            self.data_train = list(zip(torch.Tensor(enc_x_train).to(torch.long),
                                       torch.Tensor(dec_x_train).to(torch.long),
                                       torch.Tensor(dec_y_train).to(torch.long)))
            self.data_val = list(zip(torch.Tensor(enc_x_val).to(torch.long),
                                     torch.Tensor(dec_x_val).to(torch.long),
                                     torch.Tensor(dec_y_val).to(torch.long)))
            self.data_test = self.data_val

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.data_train,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.data_val,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.data_test,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=False)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.data_test,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           shuffle=False)


class ParityDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size=100,
                 n_sequences=1000,
                 min_len=10,
                 max_len=30,
                 val_split=0.2,
                 num_workers=4,
                 **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.n_sequences = n_sequences
        self.min_len = min_len
        self.max_len = max_len
        self.val_split = val_split
        self.num_workers = num_workers
        self.i_to_c_src = ['', '<START>', '<STOP>', '0', '1']
        self.i_to_c_dst = ['', '<START>', '<STOP>', '0', '1']
        self.data_train = None
        self.data_val = None
        self.data_test = None

    def collate_fn(self, batch):
        enc_x, dec_x, dec_y = zip(*batch)
        return (pad_sequence(enc_x, batch_first=True, padding_value=0),
                pad_sequence(dec_x, batch_first=True, padding_value=0),
                pad_sequence(dec_y, batch_first=True, padding_value=0))

    def setup(self, stage: str = None):
        if not (self.data_train and self.data_val and self.data_test):
            data = []
            for _ in range(self.n_sequences):
                length = random.randint(self.min_len, self.max_len)
                bits = [random.randint(0, 1) for _ in range(length)]

                cur, parity = 0, []
                for b in bits:
                    cur = (cur + b) % 2
                    parity.append(cur)

                src = [1] + [b + 3 for b in bits] + [2]
                dst = [1] + [p + 3 for p in parity] + [2]

                data.append((torch.tensor(src, dtype=torch.long),
                             torch.tensor(dst[:-1], dtype=torch.long),
                             torch.tensor(dst[1:], dtype=torch.long)))

            split_point = int(len(data) * (1.0 - self.val_split))
            self.data_train = data[:split_point]
            self.data_val = data[split_point:]
            self.data_test = self.data_val

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.data_train,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           collate_fn=self.collate_fn,
                                           shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.data_val,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           collate_fn=self.collate_fn,
                                           shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.data_test,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           collate_fn=self.collate_fn,
                                           shuffle=False)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.data_test,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers,
                                           collate_fn=self.collate_fn,
                                           shuffle=False)


class RecurrentResidual(pl.LightningModule):
    def __init__(self,
                 latent_size = 64,
                 bidirectional = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.layer_norm = torch.nn.LayerNorm(latent_size)
        hidden_size = latent_size // 2 if bidirectional else latent_size
        if USE_RNN:
            self.rnn_layer = torch.nn.RNN(latent_size,
                                          hidden_size,
                                          bidirectional=bidirectional,
                                          batch_first=True)
        else:
            self.rnn_layer = torch.nn.LSTM(latent_size,
                                           hidden_size,
                                           bidirectional=bidirectional,
                                           batch_first=True)

    def forward(self, x):
        return x + self.rnn_layer(self.layer_norm(x))[0]


class EncoderNetwork(pl.LightningModule):
    def __init__(self,
                 num_tokens,
                 latent_size = 64, # Use something divisible by 2
                 n_layers = 8,
                 bidirectional = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.embedding = torch.nn.Embedding(num_tokens,
                                            latent_size,
                                            padding_idx=0)
        self.dropout = torch.nn.Dropout1d(0.1) # Whole token dropped
        self.rnn_layers = torch.nn.Sequential(*[
            RecurrentResidual(latent_size, bidirectional) for _ in range(n_layers)
        ])

    def forward(self, x):
        y = x
        y = self.embedding(y)
        y = self.dropout(y)
        y = self.rnn_layers(y)
        y = y[torch.arange(y.shape[0], device=y.device), (x != 0).sum(1) - 1]
        return y


class DecoderNetwork(pl.LightningModule):
    def __init__(self,
                 num_tokens,
                 latent_size = 64, # Use something divisible by 2
                 n_layers = 8,
                 **kwargs):
        super().__init__(**kwargs)
        self.embedding = torch.nn.Embedding(num_tokens,
                                            latent_size,
                                            padding_idx=0)
        self.dropout = torch.nn.Dropout1d(0.1) # Whole token dropped
        self.linear = torch.nn.Linear(latent_size*2,
                                      latent_size)
        self.rnn_layers = torch.nn.Sequential(*[
            RecurrentResidual(latent_size,False) for _ in range(n_layers)
        ])
        self.output_layer = torch.nn.Linear(latent_size,
                                            num_tokens)

    def forward(self, x_enc, x_dec):
        y_enc = x_enc.unsqueeze(1).repeat(1,x_dec.shape[1],1)
        y_dec = self.embedding(x_dec)
        y_dec = self.dropout(y_dec)
        y = y_enc
        y = torch.concatenate([y_enc,y_dec],-1)
        y = self.linear(y)
        y = self.rnn_layers(y)
        y = self.output_layer(y)
        return y


# Define Trainable Module (Abstract Base Class)
class EncDecLightningBoilerplate(pl.LightningModule):
    def __init__(self, **kwargs):
        # This is the contructor, where we typically make
        # layer objects using provided arguments.
        super().__init__(**kwargs) # Call the super class constructor

    def predict_step(self, predict_batch, batch_idx):
        x_enc, x_dec, y_dec = predict_batch
        y_pred = self.predict(x_enc, x_dec)
        return y_pred, y_dec

    def training_step(self, train_batch, batch_idx):
        x_enc, x_dec, y_dec = train_batch
        y_pred = self(x_enc, x_dec)
        perm = (0,-1) + tuple(range(y_pred.ndim))[1:-1]
        for metric_name, metric_function in self.network_metrics.items():
            metric_value = metric_function(y_pred.permute(*perm),y_dec)
            self.log('train_'+metric_name, metric_value, on_step=False, on_epoch=True)
        loss = self.network_loss(y_pred.permute(*perm),y_dec)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    # Validate used for Teacher Forcing
    def validation_step(self, val_batch, batch_idx):
        x_enc, x_dec, y_dec = val_batch
        y_pred = self(x_enc, x_dec)
        perm = (0,-1) + tuple(range(y_pred.ndim))[1:-1]
        for metric_name, metric_function in self.network_metrics.items():
            metric_value = metric_function(y_pred.permute(*perm),y_dec)
            self.log('val_'+metric_name, metric_value, on_step=False, on_epoch=True)
        loss = self.network_loss(y_pred.permute(*perm),y_dec)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    # Test used for Non-Teacher Forcing
    def test_step(self, test_batch, batch_idx):
        x_enc, x_dec, y_dec = test_batch
        context = self.enc_net(x_enc)
        tokens = torch.zeros_like(x_dec).long()
        tokens[:,0] = 1
        for i in range(y_dec.shape[1]-1):
            tokens[:,i+1] = self.dec_net(context, tokens).argmax(-1)[:,i]
        y_pred = self(x_enc, tokens)
        perm = (0,-1) + tuple(range(y_pred.ndim))[1:-1]
        for metric_name, metric_function in self.network_metrics.items():
            metric_value = metric_function(y_pred.permute(*perm),y_dec)
            self.log('test_'+metric_name, metric_value, on_step=False, on_epoch=True)
        loss = self.network_loss(y_pred.permute(*perm),y_dec)
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return loss


# Attach loss, metrics, and optimizer
class EncDecLightningModule(EncDecLightningBoilerplate):
    def __init__(self,
                 num_classes,
                 ignore_index=0,
                 **kwargs):
        # This is the contructor, where we typically make
        # layer objects using provided arguments.
        super().__init__(**kwargs) # Call the super class constructor
        self.num_classes = num_classes

        # This creates an accuracy function
        self.network_metrics = torch.nn.ModuleDict({
            'acc': torchmetrics.classification.Accuracy(task='multiclass',
                                                        num_classes=num_classes,
                                                        ignore_index=ignore_index)
        })
        # This creates a loss function
        self.network_loss = torch.nn.CrossEntropyLoss(ignore_index=0)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


class EncDecNetwork(EncDecLightningModule):
    def __init__(self,
                 num_enc_tokens,
                 num_dec_tokens,
                 latent_size = 64, # Use something divisible by 2
                 n_layers = 8,
                 **kwargs):
        super().__init__(num_classes=num_dec_tokens,
                         **kwargs)
        self.enc_net = EncoderNetwork(num_enc_tokens, latent_size, n_layers)
        self.dec_net = DecoderNetwork(num_dec_tokens, latent_size, n_layers)

    def forward(self, x_enc, x_dec):
        return self.dec_net(self.enc_net(x_enc), x_dec)


def main():
    parser = argparse.ArgumentParser()

    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_CONSOLE"] = "off"
    all_acc, all_loss, all_test_acc = [], [], []

    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--data-dir', type=Path, default=HERE / 'datasets')
    parser.add_argument('--output-dir', type=Path, default=HERE)

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.output_dir / 'runs' / EXPERIMENT
    run_dir.mkdir(parents=True, exist_ok=True)

    for run in range(args.runs):

        with output_context, error_context:
            pl.seed_everything(args.seed + run, workers=True, verbose=False)

            if PARITY_PROBLEM:
                data_module = ParityDataModule(num_workers=args.workers, batch_size=args.batch_size)
            else:
                data_module = CharacterTranslationDataModule(n_sequences=None, num_workers=args.workers, batch_size=args.batch_size)

            data_module.setup('validate')

            model = EncDecNetwork(num_enc_tokens=len(data_module.i_to_c_src),
                                  num_dec_tokens=len(data_module.i_to_c_dst),
                                  n_layers=args.n_layers)

            logger = pl.loggers.CSVLogger(str(run_dir / "logs"), name=f'run-{run + 1:02d}')
            wandb_logger = pl.loggers.WandbLogger(
                project=PROJECT, name=f'{EXPERIMENT}-run{run + 1}',
                group=EXPERIMENT, save_dir=str(run_dir)
            )

            loggers = []
            loggers.append(wandb_logger)
            loggers.append(logger)
            configuration = {
                'experiment': EXPERIMENT, 'runs': args.runs, 'epochs': args.epochs,
                'seed': args.seed, 'split_seed': 0, 'batch_size': data_module.batch_size,
                'custom': USE_CUSTOM, 'data_dir': str(args.data_dir.resolve()),
            }

            configuration = json.loads(json.dumps(configuration))
            wandb_logger.log_hyperparams({**configuration, 'run': run + 1,
                                          'run_seed': args.seed + run})
            trainer = pl.Trainer(max_epochs=args.epochs, logger=loggers,
                                 enable_progress_bar=True,
                                 enable_checkpointing=False,
                                 log_every_n_steps=0,
                                 callbacks=[pl.callbacks.RichProgressBar(refresh_rate=20)])

            trainer.fit(model, data_module)
            test_results = trainer.test(model, data_module)
            wandb.finish()

        results = pd.read_csv(logger.log_dir+"/metrics.csv", delimiter=',')

        print("Validation accuracy - TF:", *["%.8f" % (x) for x in
                                             results['val_acc'][np.logical_not(np.isnan(results["val_acc"]))]])
        print("Validation loss - TF: ", *["%.8f" % (x) for x in
                                          results['val_loss'][np.logical_not(np.isnan(results["val_loss"]))]])
        print("Validation accuracy - NO TF", *["%.8f" % (x['test_acc']) for x in test_results])

        accuracy = results['val_acc'][np.logical_not(np.isnan(results['val_acc']))].to_numpy()
        loss = results['val_loss'][np.logical_not(np.isnan(results['val_loss']))].to_numpy()
        all_acc.append(accuracy)
        all_loss.append(loss)
        all_test_acc.append(test_results[0]['test_acc'])

    print("Final validation accuracy: NO TF - across all runs: "
          "%.4f +/- %.4f" % (np.mean(all_test_acc), np.std(all_test_acc)))

    for metric, curves in [('acc', all_acc), ('loss', all_loss)]:
        target = args.output_dir / f'{EXPERIMENT}-{metric}-results.txt'
        np.savetxt(target, np.array(curves), fmt='%.8f')


if __name__ == '__main__':
    main()
