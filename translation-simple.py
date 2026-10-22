import numpy as np
import torch
import lightning.pytorch as pl
import torchmetrics
import torchvision
from torchinfo import summary
from torchview import draw_graph
import pandas as pd
import matplotlib.pyplot as plt

USE_RNN = True
USE_CUSTOM = False
PARITY_PROBLEM = False

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

data_module = CharacterTranslationDataModule(n_sequences=None)
data_module.setup('validate')
dl = data_module.val_dataloader()

class RecurrentResidual(pl.LightningModule):
    def __init__(self,
                 latent_size = 64,
                 bidirectional = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.layer_norm = torch.nn.LayerNorm(latent_size)
        self.rnn_layer = torch.nn.RNN(latent_size, latent_size, batch_first=True)
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


# enc_net = EncoderNetwork(num_tokens=len(data_module.i_to_c_src))

batch = next(iter(dl))

# summary(enc_net,
#         input_size=batch[0].shape,
#         dtypes=[torch.long])

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

# dec_net = DecoderNetwork(num_tokens=len(data_module.i_to_c_dst))
#
# summary(dec_net,input_data=[enc_net(batch[0].to(enc_net.device)).to(dec_net.device),
#                             batch[1]])

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
        self.enc_net = EncoderNetwork(num_enc_tokens,latent_size,n_layers)
        self.dec_net = DecoderNetwork(num_dec_tokens,latent_size,n_layers)

    def forward(self, x_enc, x_dec):
        return self.dec_net(self.enc_net(x_enc), x_dec)

enc_dec_net = EncDecNetwork(num_enc_tokens=len(data_module.i_to_c_src),
                            num_dec_tokens=len(data_module.i_to_c_dst))
summary(enc_dec_net,input_data=[batch[0].to(enc_dec_net.device),
                                batch[1].to(enc_dec_net.device)])

logger = pl.loggers.CSVLogger("logs",
                              name="Demo-10",
                              version="character-eng-to-por")

trainer = pl.Trainer(logger=logger,
                     max_epochs=300,
                     enable_progress_bar=True,
                     enable_checkpointing=False,
                     log_every_n_steps=0,
                     callbacks=[pl.callbacks.RichProgressBar(refresh_rate=50)])

trainer.fit(enc_dec_net, data_module)

trainer.test(enc_dec_net, data_module)