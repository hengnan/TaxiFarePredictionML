import sys
import json
import time
import torch as pt
import pytorch_lightning as pl
from distutils.util import strtobool

pt.set_default_dtype(pt.float64)


class DcTaxiModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        pt.manual_seed(int(self.hparams.seed))

        self.step = 0
        self.start_ts = time.perf_counter()
        self.train_val_rmse = pt.tensor(0.)

        #create a list of hidden layer neurons, e.g. [3, 5, 8]
        num_hidden_neurons = json.loads(self.hparams.num_hidden_neurons)

        self.layers = \
            pt.nn.Sequential(
                pt.nn.Linear(int(self.hparams.num_features),
                             num_hidden_neurons[0]),
                pt.nn.ReLU(),
                *self.build_hidden_layers(num_hidden_neurons, pt.nn.ReLU()),
                pt.nn.Linear(num_hidden_neurons[-1], 1)
            )

        if 'batch_norm_linear_layers' in self.hparams \
                and strtobool(self.hparams.batch_norm_linear_layers):
            self.layers = self.batch_norm_linear(self.layers)

    def build_hidden_layers(self, num_hidden_neurons, activation):
        linear_layers = [pt.nn.Linear(num_hidden_neurons[i],
                                      num_hidden_neurons[i + 1]) \
                         for i in range(len(num_hidden_neurons) - 1)]

        classes = [activation.__class__] * len(num_hidden_neurons)

        activation_instances = list(map(lambda x: x(), classes))

        hidden_layer_activation_tuples = \
            list(zip(linear_layers, activation_instances))

        hidden_layers = [i for sublist in \
                         hidden_layer_activation_tuples for i in sublist]

        return hidden_layers

    def batch_norm_linear(self, layers):
        idx_linear = \
            list(filter(lambda x: type(x) is int,
                        [idx if issubclass(layer.__class__, pt.nn.Linear) else None \
                         for idx, layer in enumerate(layers)]))
        idx_linear.append(sys.maxsize)
        layer_lists = [list(iter(layers[s:e])) \
                       for s, e in zip(idx_linear[:-1], idx_linear[1:])]
        batch_norm_layers = [pt.nn.BatchNorm1d(layer[0].in_features) \
                             for layer in layer_lists]
        batch_normed_layer_lists = [[bn, *layers] \
                                    for bn, layers in list(zip(batch_norm_layers, layer_lists))]
        return pt.nn.Sequential(*[layer \
                                  for nested_layer in batch_normed_layer_lists \
                                  for layer in nested_layer])

    def batchToXy(self, batch):
        batch = batch.squeeze_()
        X, y = batch[:, 1:], batch[:, 0]
        return X, y

    def forward(self, X):
        y_est = self.layers(X)
        return y_est.squeeze_()

    def log(self, k, v, **kwargs):
        super().log(k, v,
                    on_step=kwargs['on_step'],
                    on_epoch=kwargs['on_epoch'],
                    prog_bar=kwargs['prog_bar'],
                    logger=kwargs['logger'], )

    def training_step(self, batch, batch_idx):
        self.step += 1

        X, y = self.batchToXy(batch)  #unpack batch into features and label

        y_est = self.forward(X)

        loss = pt.nn.functional.mse_loss(y_est, y)

        for k, v in {
            "train_step": self.step,
            "train_mse": loss.item(),
            "train_rmse": loss.sqrt().item(),
            "train_steps_per_sec": \
                    self.step / (time.perf_counter() - self.start_ts),

        }.items():
            self.log(k, v, step=self.step, on_step=True, on_epoch=True,
                     prog_bar=True, logger=True)

        self.train_val_rmse = loss.sqrt()

        return loss

    def validation_step(self, batch, batch_idx):
        X, y = self.batchToXy(batch)

        with pt.no_grad():
            loss = pt.nn.functional.mse_loss(self.forward(X), y)

        for k, v in {
            "val_mse": loss.item(),
            "val_rmse": loss.sqrt().item(),
            "train_val_rmse": (self.train_val_rmse + loss.sqrt()).item(),
        }.items():
            self.log(k, v, step=self.step, on_step=True, on_epoch=True,
                     prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        X, y = self.batchToXy(batch)

        with pt.no_grad():
            loss = pt.nn.functional.mse_loss(self.forward(X), y)

        for k, v in {
            "test_mse": loss.item(),
            "test_rmse": loss.sqrt().item(),
        }.items():
            self.log(k, v, step=self.step, on_step=True, on_epoch=True,
                     prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizers = {'Adam': pt.optim.AdamW,
                      'SGD': pt.optim.SGD}
        optimizer = optimizers[self.hparams.optimizer]

        return optimizer(self.layers.parameters(),
                         lr=float(self.hparams.lr))
