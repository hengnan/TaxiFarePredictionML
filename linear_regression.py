import os
import time
import torch as pt
import fsspec

fsspec.core.DEFAULT_EXPAND = True
from torch.utils.data import TensorDataset, DataLoader
from kaen.torch import ObjectStorageDataset as osds

pt.manual_seed(0);
pt.set_default_dtype(pt.float64)

BUCKET_ID = os.environ['BUCKET_ID']
AWS_DEFAULT_REGION = os.environ['AWS_DEFAULT_REGION']

BATCH_SIZE = 2 ** 20
train_ds = osds(f"s3://dc-taxi-{BUCKET_ID}-{AWS_DEFAULT_REGION}/csv/dev/part*.csv",
                storage_options={'anon': False},
                batch_size=BATCH_SIZE)

train_dl = DataLoader(train_ds, batch_size=None)

FEATURE_COUNT = 8

w = pt.nn.init.kaiming_uniform_(pt.empty(FEATURE_COUNT,
                                         1, requires_grad=True))
b = pt.nn.init.kaiming_uniform_(pt.empty(1,
                                         1, requires_grad=True))


def batchToXy(batch):
    batch = batch.squeeze_()
    return batch[:, 1:], batch[:, 0]


def forward(X):
    y_est = X @ w + b
    return y_est.squeeze_()


LEARNING_RATE = 0.03
optimizer = pt.optim.SGD([w, b], lr=LEARNING_RATE)

GRADIENT_NORM = None

ITERATION_COUNT = 5

for iter_idx, batch in zip(range(ITERATION_COUNT), train_dl):
    start_ts = time.perf_counter()

    X, y = batchToXy(batch)

    y_est = forward(X)
    mse = pt.nn.functional.mse_loss(y_est, y)
    mse.backward()

    pt.nn.utils.clip_grad_norm_([w, b],
                                GRADIENT_NORM) if GRADIENT_NORM else None

    optimizer.step()
    optimizer.zero_grad()

    sec_iter = time.perf_counter() - start_ts

    print(f"Iteration: {iter_idx:03d}, Seconds/Iteration: {sec_iter:.3f}MSE: {mse.data.item(): .2f}")
