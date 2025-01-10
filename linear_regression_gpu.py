import os
import time

import torch as pt
from torch.utils.data import DataLoader
from kaen.torch import ObjectStorageDataset as osds

pt.manual_seed(0);
pt.set_default_dtype(pt.float64)

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

BATCH_SIZE = 1_048_576  # = 2 ** 20

train_ds = osds(f"s3://dc-taxi-{os.environ['BUCKET_ID']}-{os.environ['AWS_DEFAULT_REGION']}/csv/dev/part*.csv",
                storage_options={'anon': False},
                batch_size=BATCH_SIZE)

train_dl = DataLoader(train_ds,
                      pin_memory=True)

FEATURE_COUNT = 8
w = pt.nn.init.kaiming_uniform_(pt.empty(FEATURE_COUNT, 1,
                                         requires_grad=True, device=device))
b = pt.nn.init.kaiming_uniform_(pt.empty(1, 1,
                                         requires_grad=True, device=device))


def batchToXy(batch):
    batch = batch.squeeze_().to(device)
    return batch[:, 1:], batch[:, 0]


def forward(X):
    y_pred = X @ w + b
    return y_pred.squeeze_()


def loss(y_est, y):
    mse_loss = pt.mean((y_est - y) ** 2)
    return mse_loss


LEARNING_RATE = 0.03
optimizer = pt.optim.SGD([w, b], lr=LEARNING_RATE)

GRADIENT_NORM = 0.5

ITERATION_COUNT = 50

for iter_idx, batch in zip(range(ITERATION_COUNT), train_dl):
    start_ts = time.perf_counter()

    X, y = batchToXy(batch)

    y_est = forward(X)
    mse = loss(y_est, y)
    mse.backward()

    pt.nn.utils.clip_grad_norm_([w, b],
                                GRADIENT_NORM) if GRADIENT_NORM else None

    optimizer.step()
    optimizer.zero_grad()

    sec_iter = time.perf_counter() - start_ts

    print(f"Iteration: {iter_idx:03d}, Seconds/Iteration: {sec_iter:.3f} MSE: {mse.data.item(): .2f}")
