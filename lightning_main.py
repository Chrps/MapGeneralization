import argparse
import os
import torch.nn.functional as F

import torch
import pytorch_lightning as pl


hparams = {'train_list': 'data/public_val.txt',
           'test_list': 'data/public_test.txt',
           'val_list': 'data/public_val.txt',
           'network': 'gat', #gcn', 'gat', 'monet', 'graphsage', 'appnp', 'tagcn', 'agnn', 'sgc', 'gin', 'chebnet'
           'extra_args': [12, 10, [3] * 12 + [1], F.elu, 0.1, 0.1, 0.2, True],
           'lr': 1e-3,
           'batch_size': 5,
           'n_classes': 2,
           'n_features': 3,
           'windowing': False,
           'n_workers':0}


if __name__ == '__main__':
    """
    Trains a patch classifier.

    Command:
        python main.py
    """

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--fine", type=str,
                    default='fine', help="Flag")

    args = vars(ap.parse_args())

    logger = pl.loggers.TensorBoardLogger('lightning_logs/cpu-net-{}_bs-{}'.format(hparams['network'],hparams['batch_size']))
    trainer = pl.Trainer(gpus=None, max_epochs=1000, logger=logger) #gpus=None

    from lightning_model import LightningNodeClassifier
    model = LightningNodeClassifier(hparams)

    trainer.fit(model)
    #torch.save(model.model, "pymodel.pt")

    #trainer.test(model)
