from model_v1 import DcTaxiModel

import os
import time
import kaen
import torch as pt
import numpy as np
import pytorch_lightning as pl
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from kaen.torch import ObjectStorageDataset as osds


def train(model, train_glob, val_glob, test_glob=None):
    # set the pseudorandom number generator seed
    seed = int(model.hparams['seed']) \
        if 'seed' in model.hparams \
        else int(datetime.now().microsecond)

    np.random.seed(seed)
    pt.manual_seed(seed)

    kaen.torch.init_process_group(model.layers)

    trainer = pl.Trainer(num_nodes=pt.cuda.device_count() \
         if pt.cuda.is_available() else 0,
         max_epochs=1,
         limit_train_batches=int(model.hparams.max_batches) \
             if 'max_batches' in model.hparams else 1,
         limit_val_batches=1,
         num_sanity_val_steps=1,
         val_check_interval=min(20, int(model.hparams.max_batches)),
         limit_test_batches=1,
         log_every_n_steps=1,
         gradient_clip_val=0.5,
         enable_progress_bar=True,  # Replaces `progress_bar_refresh_rate`
         enable_model_summary=True,)

    train_dl = \
        DataLoader(osds(train_glob,
        worker=kaen.torch.get_worker_rank(),
        replicas=kaen.torch.get_num_replicas(),
        shard_size= \
            int(model.hparams.batch_size),
        batch_size= \
            int(model.hparams.batch_size),
        storage_options={'anon': False},
        ),
        pin_memory=True)

    val_dl = \
        DataLoader(osds(val_glob,
        batch_size=int(model.hparams.batch_size),
        storage_options={'anon': False},
        ),
        pin_memory=True)

    trainer.fit(model,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl)
    if test_glob is not None:
        test_dl = \
            DataLoader(osds(test_glob,
            batch_size=int(model.hparams.batch_size),
            storage_options={'anon': False},
            ),
            pin_memory=True)

        trainer.test(model,
                     dataloaders=test_dl)

    return model, trainer


if __name__ == "__main__":
    model, trainer = train(DcTaxiModel(**{
        "seed": "1686523060",
        "num_features": "8",
        "num_hidden_neurons": "[3, 5, 8]",
        "batch_norm_linear_layers": "1",
        "optimizer": "Adam",
        "lr": "0.03",
        "max_batches": "1",
        "batch_size": str(2 ** 18), }),

        train_glob= \
           os.environ['KAEN_OSDS_TRAIN_GLOB'] \
               if 'KAEN_OSDS_TRAIN_GLOB' in os.environ \
               else 'https://raw.githubusercontent.com/osipov/smlbook/master/train.csv',

        val_glob = \
            os.environ['KAEN_OSDS_VAL_GLOB'] \
                if 'KAEN_OSDS_VAL_GLOB' in os.environ \
                else 'https://raw.githubusercontent.com/osipov/smlbook/master/valid.csv',

        test_glob = \
            os.environ['KAEN_OSDS_TEST_GLOB'] \
                if 'KAEN_OSDS_TEST_GLOB' in os.environ \
                else 'https://raw.githubusercontent.com/osipov/smlbook/master/valid.csv')

    print(trainer.callback_metrics)