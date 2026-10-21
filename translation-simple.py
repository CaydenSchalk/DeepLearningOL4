import numpy as np
import torch
import lightning.pytorch as pl
import torchmetrics
import torchvision
from torchinfo import summary
from torchview import draw_graph
import pandas as pd
import matplotlib.pyplot as plt

torch.set_float32_matmul_precision('medium')

print("GPU is present:",torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:",torch.cuda.get_device_name())
    print("GPU Count:",torch.cuda.device_count())
    print("GPU Properties:",torch.cuda.get_device_properties())

import urllib


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

data_module = CharacterTranslationDataModule()
data_module.setup('validate')
dl = data_module.val_dataloader()

class RecurrentResidual(pl.LightningModule):
    def __init__(self,
                 latent_size = 64,
                 bidirectional = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.layer_norm = torch.nn.LayerNorm(latent_size)
        self.rnn_layer = torch.nn.LSTM(latent_size,
                                       latent_size // 2 if bidirectional else latent_size,
                                       bidirectional=bidirectional,
                                       batch_first=True)
    def forward(self, x):
        return x + self.rnn_layer(self.layer_norm(x))[0]


class EncoderNetwork(pl.LightningModule):
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
        self.rnn_layers = torch.nn.Sequential(*[
            RecurrentResidual(latent_size,True) for _ in range(n_layers)
        ])

    def forward(self, x):
        y = x
        y = self.embedding(y)
        y = self.dropout(y)
        y = self.rnn_layers(y)[:,-1]
        return y


enc_net = EncoderNetwork(num_tokens=len(data_module.i_to_c_src))

batch = next(iter(dl))

summary(enc_net,
        input_size=batch[0].shape,
        dtypes=[torch.long])